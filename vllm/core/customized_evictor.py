import heapq
from typing import Dict, List, Tuple
from vllm.core.evictor import BlockMetaData
from vllm.core.evictor import Evictor
from collections import OrderedDict, deque
from typing import Dict, List, Tuple

class Customized2QEvictor(Evictor):

    def __init__(self, max_size: int=134, k: int=50):
        self.max_size = max_size
        self.k = k
        self.A1in: OrderedDict[int, BlockMetaData] = OrderedDict()
        self.Am: OrderedDict[int, BlockMetaData] = OrderedDict()
        self.A1out: Deque[int] = deque(maxlen=self.k)

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.A1in or block_id in self.Am

    def evict(self) -> Tuple[int, int]:
        # print('[EVICT] len(self.A1in, A1out, Am):', len(self.A1in), len(self.A1out), len(self.Am))
        if len(self.A1in) > self.k:
            block_id, meta = self.A1in.popitem(last=False)
            self.A1out.append(block_id)
            return block_id, meta.content_hash
        elif len(self.Am) > self.max_size - self.k:
            block_id, meta = self.Am.popitem(last=False)
            return block_id, meta.content_hash
        elif self.A1in:
            block_id, meta = self.A1in.popitem(last=False)
            self.A1out.append(block_id)
            return block_id, meta.content_hash
        elif self.Am:
            block_id, meta = self.Am.popitem(last=False)
            return block_id, meta.content_hash
        raise RuntimeError("No block available to evict")
    
    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        # print('[ADD] len(self.A1in, A1out, Am):', len(self.A1in), len(self.A1out), len(self.Am))
        meta = BlockMetaData(content_hash, num_hashed_tokens, last_accessed)

        if block_id in self.Am:     # Already in Am → refresh LRU position
            self.Am.move_to_end(block_id)
            return
        if block_id in self.A1in:   # Already in A1in → no-op
            self.A1in.pop(block_id)
            self._add_to_Am(block_id, meta)
            return
        if block_id in self.A1out:  # Ghost hit → promote to Am
            self.A1out.remove(block_id)
            self._add_to_Am(block_id, meta)
            return
        self.A1in[block_id] = meta  # First time → insert into A1in
        self.A1in.move_to_end(block_id)

    def update(self, block_id: int, last_accessed: float):
        assert False
        if block_id in self.A1in:
            self.A1in[block_id].last_accessed = last_accessed
        elif block_id in self.Am:
            self.Am[block_id].last_accessed = last_accessed
            self.Am.move_to_end(block_id)

    def remove(self, block_id: int):
        if block_id in self.A1in:
            self.A1in.pop(block_id)
        elif block_id in self.Am:
            # assert False
            self.Am.pop(block_id)
        else:
            raise ValueError(f"Block {block_id} not tracked")
        

    @property
    def num_blocks(self) -> int:
        return len(self.A1in) + len(self.Am)
    
    def _add_to_Am(self, block_id: int, meta: BlockMetaData):
        self.Am[block_id] = meta
        self.Am.move_to_end(block_id)


class CustomizedDBLEvictor(Evictor):
    CLEANUP_THRESHOLD = 50  # Threshold to trigger heap cleanup

    def __init__(self, max_size: int = 134, k: int = 67):
        self.max_size = max_size
        self.k = k

        # Free tables for active entries
        self.A1in_free_table: Dict[int, BlockMetaData] = {}
        self.Am_free_table: Dict[int, BlockMetaData] = {}

        # Priority queues for eviction order: (last_accessed, -tokens, block_id, content_hash)
        self.A1in_priority_queue: List[Tuple[float, int, int, int]] = []
        self.Am_priority_queue: List[Tuple[float, int, int, int]] = []

        # Ghost queues for tracking evicted entries
        self.A1out_ghost: deque[int] = deque(maxlen=self.max_size * 100)
        self.Amout_ghost: deque[int] = deque(maxlen=self.max_size * 100)

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.A1in_free_table or block_id in self.Am_free_table

    def evict(self, content_hash: int = None) -> Tuple[int, int]:
        # Always evict from A1in if available, else from Am
        if len(self.A1in_free_table) > self.k:
            return self._evict_from_A1in()
        elif len(self.Am_free_table) > self.max_size - self.k:
            return self._evict_from_Am()
        elif self.A1in_free_table:
            return self._evict_from_A1in()
        if self.Am_free_table:
            return self._evict_from_Am()
        raise ValueError("No usable cache memory left")

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int, last_accessed: float):
        """
        Register a block for future eviction:
        - No eviction should occur here; eviction is driven externally.
        - Blocks must not already exist in A1in or Am.
        """
        # print(f"DBLCache A1: {len(self.A1in_free_table)}, Am: {len(self.Am_free_table)}")
        # Ensure this is a new registration
        assert block_id not in self.A1in_free_table and block_id not in self.Am_free_table, \
            f"Block {block_id} unexpectedly already tracked"

        meta = BlockMetaData(content_hash, num_hashed_tokens, last_accessed)

        # assert block_id in self.A1out_ghost or block_id in self.Amout_ghost
        # Promote from A1out ghost to Am if ghost hit
        if block_id in self.A1out_ghost:
            self.A1out_ghost.remove(block_id)
            self.Am_free_table[block_id] = meta
            heapq.heappush(
                self.Am_priority_queue,
                (last_accessed, -num_hashed_tokens, block_id, content_hash)
            )
            self._cleanup_if_necessary(self.Am_priority_queue, self.Am_free_table)
            return

        # Promote from Amout ghost to Am if ghost hit
        if block_id in self.Amout_ghost:
            self.Amout_ghost.remove(block_id)
            self.Am_free_table[block_id] = meta
            heapq.heappush(
                self.Am_priority_queue,
                (last_accessed, -num_hashed_tokens, block_id, content_hash)
            )
            self._cleanup_if_necessary(self.Am_priority_queue, self.Am_free_table)
            return

        # Insert into A1in for new block
        self.A1in_free_table[block_id] = meta
        heapq.heappush(
            self.A1in_priority_queue,
            (last_accessed, -num_hashed_tokens, block_id, content_hash)
        )
        self._cleanup_if_necessary(self.A1in_priority_queue, self.A1in_free_table)

    def update(self, block_id: int, last_accessed: float):
        # Only update last_accessed in free tables; add will not see existing blocks
        if block_id in self.A1in_free_table:
            self.A1in_free_table[block_id].last_accessed = last_accessed
        elif block_id in self.Am_free_table:
            self.Am_free_table[block_id].last_accessed = last_accessed
        else:
            raise ValueError(f"Attempting to update non-tracked block {block_id}")

    def remove(self, block_id: int):
        if block_id in self.A1in_free_table:
            self.A1in_free_table.pop(block_id)
            self.A1out_ghost.append(block_id)
        elif block_id in self.Am_free_table:
            self.Am_free_table.pop(block_id)
            self.Amout_ghost.append(block_id)
        else:
            raise ValueError(f"Attempting to remove non-tracked block {block_id}")

    @property
    def num_blocks(self) -> int:
        return len(self.A1in_free_table) + len(self.Am_free_table)

    def _evict_from_A1in(self) -> Tuple[int, int]:
        # Evict oldest from A1in and record to ghost
        while self.A1in_priority_queue:
            last_accessed, neg_tokens, block_id, content_hash = heapq.heappop(self.A1in_priority_queue)
            if (block_id in self.A1in_free_table and
                    self.A1in_free_table[block_id].last_accessed == last_accessed):
                self.A1in_free_table.pop(block_id)
                # self.A1out_ghost.append(block_id)
                return block_id, content_hash
        raise ValueError("No usable block left in A1in to evict")

    def _evict_from_Am(self) -> Tuple[int, int]:
        # Evict oldest from Am and record to ghost
        while self.Am_priority_queue:
            last_accessed, neg_tokens, block_id, content_hash = heapq.heappop(self.Am_priority_queue)
            if (block_id in self.Am_free_table and
                    self.Am_free_table[block_id].last_accessed == last_accessed):
                self.Am_free_table.pop(block_id)
                # self.Amout_ghost.append(block_id)
                return block_id, content_hash
        raise ValueError("No usable block left in Am to evict")

    def _cleanup_if_necessary(self, queue: List[Tuple[float, int, int, int]], table: Dict[int, BlockMetaData]):
        if len(queue) > self.CLEANUP_THRESHOLD * len(table):
            self._cleanup(queue, table)

    def _cleanup(self, queue: List[Tuple[float, int, int, int]], table: Dict[int, BlockMetaData]):
        new_queue = []
        for block_id, meta in table.items():
            new_queue.append((meta.last_accessed, -meta.num_hashed_tokens, block_id, meta.content_hash))
        heapq.heapify(new_queue)

        if table is self.A1in_free_table:
            self.A1in_priority_queue = new_queue
        else:
            self.Am_priority_queue = new_queue


class CustomizedARCEvictor(Evictor):
    CLEANUP_THRESHOLD = 50  # Threshold to trigger heap cleanup

    def __init__(self, max_size: int = 134):
        self.max_size = max_size
        self.p = 0  # ARC 中对 T1 的目标大小

        # 活跃区 free tables
        self.T1_free_table: Dict[int, BlockMetaData] = {}
        self.T2_free_table: Dict[int, BlockMetaData] = {}

        # 对应的优先队列 (last_accessed, -tokens, block_id, content_hash)
        self.T1_priority_queue: List[Tuple[float,int,int,int]] = []
        self.T2_priority_queue: List[Tuple[float,int,int,int]] = []

        # 幽灵区
        self.B1_ghost: deque[int] = deque()     # Record content_hash
        self.B2_ghost: deque[int] = deque()

        # 记录外部 remove 的 block id，用于下次 add 时直接晋升
        self.removed_T1: deque[int] = deque()   # Record block_id
        self.removed_T2: deque[int] = deque()
        
        self.removed_B1: deque[int] = deque()   # Record content_hash
        self.removed_B2: deque[int] = deque()
        
        self.evicted_T1: deque[int] = deque()   # Record block_id
        self.evicted_T2: deque[int] = deque()


    def __contains__(self, block_id: int) -> bool:
        return (block_id in self.T1_free_table) or (block_id in self.T2_free_table)

    def evict(self, content_hash: int = None) -> Tuple[int, int]:
        # print('len T1, T2', len(self.T1_free_table), len(self.T2_free_table), 'evict for:', content_hash)
        assert content_hash not in self.T1_free_table and content_hash not in self.T2_free_table
        # modify p
        if content_hash in self.B1_ghost:
            delta = max(1, len(self.B2_ghost) // max(1, len(self.B1_ghost)))
            self.p = min(self.p + delta, self.max_size)
        elif content_hash in self.B2_ghost:
            delta = max(1, len(self.B1_ghost) // max(1, len(self.B2_ghost)))
            self.p = max(self.p - delta, 0)
        
        # Choose the Evicted Candidate
        evicted_block_id: int = -1
        evicted_content_hash: int = -1
        L1_size = len(self.T1_free_table) + len(self.removed_T1) + len(self.evicted_T1) + len(self.B1_ghost)
        if content_hash not in self.B1_ghost and content_hash not in self.B2_ghost and len(self.B1_ghost) == 0 and L1_size >= self.max_size:
        # if content_hash not in self.B1_ghost and content_hash not in self.B2_ghost and len(self.T2_free_table) == 0:
            evicted_block_id, evicted_content_hash = self._evict_from_T1()
        else:
            evicted_block_id, evicted_content_hash = self._replace(content_hash)
        # print('evicted_content_hash:', evicted_content_hash)
            
        # Delete for_block from B1/B2 and record it for future add()
        # print(f'{content_hash} in B1 B2', content_hash in self.B1_ghost, content_hash in self.B2_ghost)

        if content_hash in self.B1_ghost:
            print('B1 hit', L1_size)
            self.B1_ghost.remove(content_hash)
            # print(f'{content_hash} in B1 after remove:', content_hash in self.B1_ghost)
            self.removed_B1.append(content_hash)
        elif content_hash in self.B2_ghost:
            print('B2 hit', L1_size)
            self.B2_ghost.remove(content_hash)
            self.removed_B2.append(content_hash)
        else:
            L1_size = len(self.T1_free_table) + len(self.evicted_T1) + len(self.B1_ghost)
            L2_size = len(self.T2_free_table) + len(self.removed_T1) + len(self.removed_T2) + len(self.evicted_T2) + len(self.B2_ghost)
            print('B miss, L1_size T1_total B1:', L1_size, len(self.T1_free_table) + len(self.removed_T1), len(self.B1_ghost))
            total_size = L1_size + L2_size
            # if L1_size >= self.max_size and len(self.T1_free_table) + len(self.removed_T1) < self.max_size and not self.B1_ghost:
            if L1_size >= self.max_size and self.B1_ghost:
                print('B miss, popleft()')
                self.B1_ghost.popleft()
            elif L1_size < self.max_size and total_size > self.max_size:
                if total_size == 2 * self.max_size and self.B2_ghost:
                    self.B2_ghost.popleft()
        
        return evicted_block_id, evicted_content_hash
        

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int, last_accessed: float):
        print(f'add - len T1, T2, B1, B2 | {len(self.T1_free_table)} ({len(self.removed_T1)}) {len(self.T2_free_table)} ({len(self.removed_T2)})  | {len(self.B1_ghost)} ({len(self.removed_B1)}) {len(self.B2_ghost)} ({len(self.removed_B2)})  | p: {self.p} content_hash: {content_hash}')
        # assert block_id not in self.T1_free_table and block_id not in self.T2_free_table
        meta = BlockMetaData(content_hash, num_hashed_tokens, last_accessed)
        
        if block_id in self.evicted_T1:
            self.evicted_T1.remove(block_id)
        if block_id in self.evicted_T2:
            self.evicted_T2.remove(block_id)
        # Removed block: promote to T2
        if block_id in self.removed_T1 or block_id in self.removed_T2:
            if block_id in self.removed_T1:
                self.removed_T1.remove(block_id)
            if block_id in self.removed_T2:
                self.removed_T2.remove(block_id)
            self.T2_free_table[block_id] = meta
            heapq.heappush(
                self.T2_priority_queue,
                (last_accessed, -num_hashed_tokens, block_id, content_hash)
            )
            self._cleanup_if_necessary(self.T2_priority_queue, self.T2_free_table)
            return

        # print(f"{content_hash} in B1 or B2:", content_hash in self.B1_ghost, content_hash in self.B2_ghost)
        # assert content_hash not in self.B1_ghost and content_hash not in self.B2_ghost
        # Ghost hit in B1: promote to T2 and increase p
        if content_hash in self.removed_B1:
            self.removed_B1.remove(content_hash)
            self.T2_free_table[block_id] = meta
            heapq.heappush(self.T2_priority_queue, (last_accessed, -num_hashed_tokens, block_id, content_hash))
            self._cleanup_if_necessary(self.T2_priority_queue, self.T2_free_table)
            return
        
        # Ghost hit in B2: promote to T2 and decrease p
        if content_hash in self.removed_B2:
            self.removed_B2.remove(content_hash)
            self.T2_free_table[block_id] = meta
            heapq.heappush(self.T2_priority_queue, (last_accessed, -num_hashed_tokens, block_id, content_hash))
            self._cleanup_if_necessary(self.T2_priority_queue, self.T2_free_table)
            return
        
        # New block: insert into T1
        self.T1_free_table[block_id] = meta
        heapq.heappush(self.T1_priority_queue, (last_accessed, -num_hashed_tokens, block_id, content_hash))
        self._cleanup_if_necessary(self.T1_priority_queue, self.T1_free_table)
        self._prune_ghosts()

    def update(self, block_id: int, last_accessed: float):
        assert False
        # Update last_accessed for an already-tracked block
        if block_id in self.T1_free_table:
            self.T1_free_table[block_id].last_accessed = last_accessed
        elif block_id in self.T2_free_table:
            self.T2_free_table[block_id].last_accessed = last_accessed
        else:
            raise ValueError(f"Attempting to update non-tracked block {block_id}")

    def remove(self, block_id: int):
        if block_id in self.T1_free_table:
            content_hash = self.T1_free_table[block_id].content_hash
            # print('remove hit T1:', content_hash)
            # assert content_hash not in self.B1_ghost
            
            self.T1_free_table.pop(block_id)
            self.removed_T1.append(block_id)
        elif block_id in self.T2_free_table:
            content_hash = self.T2_free_table[block_id].content_hash
            # print('remove hit T2:', content_hash)
            # assert content_hash not in self.B2_ghost
            
            self.T2_free_table.pop(block_id)
            self.removed_T2.append(block_id)
        else:
            raise ValueError(f"Attempting to remove non-tracked block {block_id}")

    @property
    def num_blocks(self) -> int:
        return len(self.T1_free_table) + len(self.T2_free_table)
    
    def _replace(self, content_hash_replace_for: int):
        Len_cached_T1 = len(self.T1_free_table) + len(self.evicted_T1)
        if self.T1_free_table and (
            (content_hash_replace_for in self.B2_ghost and Len_cached_T1 == self.p) or
            (Len_cached_T1 > self.p)
        ):
            block_id, content_hash = self._evict_from_T1()
            self.B1_ghost.append(content_hash)
            return block_id, content_hash
        elif self.T2_free_table:
            block_id, content_hash = self._evict_from_T2()
            self.B2_ghost.append(content_hash)
            return block_id, content_hash
        raise ValueError("No usable block to replace from T1 or T2")

    def _evict_from_T1(self) -> Tuple[int, int]:
        while self.T1_priority_queue:
            last_accessed, neg_tokens, block_id, content_hash = heapq.heappop(self.T1_priority_queue)
            if (block_id in self.T1_free_table and
                    self.T1_free_table[block_id].last_accessed == last_accessed):
                self.T1_free_table.pop(block_id)
                self.evicted_T1.append(block_id)
                return block_id, content_hash
        raise ValueError("No usable block left in T1 to evict")

    def _evict_from_T2(self) -> Tuple[int, int]:
        while self.T2_priority_queue:
            last_accessed, neg_tokens, block_id, content_hash = heapq.heappop(self.T2_priority_queue)
            if (block_id in self.T2_free_table and
                    self.T2_free_table[block_id].last_accessed == last_accessed):
                self.T2_free_table.pop(block_id)
                self.evicted_T2.append(block_id)
                return block_id, content_hash
        raise ValueError("No usable block left in T2 to evict")
    
    def _prune_ghosts(self):
        """确保 ghost 列表 B1_ghost 和 B2_ghost 的大小不超过 max_size"""
        assert len(self.removed_B1) <= self.max_size and len(self.removed_B2) <= self.max_size
        while len(self.B1_ghost) + len(self.removed_B1) > self.max_size:
            self.B1_ghost.popleft()
        while len(self.B2_ghost) + len(self.removed_B2) > self.max_size:
            self.B2_ghost.popleft()


    def _cleanup_if_necessary(self, queue: List[Tuple[float, int, int, int]], table: Dict[int, BlockMetaData]):
        if len(queue) > self.CLEANUP_THRESHOLD * len(table):
            self._cleanup(queue, table)

    def _cleanup(self, queue: List[Tuple[float, int, int, int]], table: Dict[int, BlockMetaData]):
        new_queue = []
        for block_id, meta in table.items():
            new_queue.append((meta.last_accessed, -meta.num_hashed_tokens, block_id, meta.content_hash))
        heapq.heapify(new_queue)

        if queue is self.T1_priority_queue:
            self.T1_priority_queue = new_queue
        else:
            self.T2_priority_queue = new_queue
            

class CustomizedLRUEvictor(Evictor):

    def __init__(self):
        self.cache: [int, BlockMetaData] = OrderedDict()

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.cache

    def evict(self) -> Tuple[int, int]:
        # print('[EVICT] len(self.A1in, A1out, Am):', len(self.A1in), len(self.A1out), len(self.Am))
        if self.cache:
            block_id, meta = self.cache.popitem(last=False)
            return block_id, meta.content_hash
        raise RuntimeError("No block available to evict")
    
    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        # print('[ADD] len(self.A1in, A1out, Am):', len(self.A1in), len(self.A1out), len(self.Am))
        assert block_id not in self.cache
        meta = BlockMetaData(content_hash, num_hashed_tokens, last_accessed)
        self.cache[block_id] = meta
        self.cache.move_to_end(block_id)
        
    def update(self, block_id: int, last_accessed: float):
        assert False
        if block_id in self.cache:
            self.cache[block_id].last_accessed = last_accessed

    def remove(self, block_id: int):
        if block_id in self.cache:
            self.cache.pop(block_id)
        else:
            raise ValueError(f"Block {block_id} not tracked")
        
    @property
    def num_blocks(self) -> int:
        return len(self.cache)
