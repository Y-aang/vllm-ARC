import heapq
from typing import Dict, List, Tuple, Optional, Set, Deque
from vllm.core.evictor import BlockMetaData
from vllm.core.evictor import Evictor
from vllm.core.customized_evictor_utils import TailTable
from collections import OrderedDict, deque
import time     # For Debug

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
            last_accessed: float, prev_block_content_hash: Optional[int] = None):
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

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int, 
            last_accessed: float, prev_block_content_hash: Optional[int] = None):
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
        self.p = 0  # ARC's adjustment coefficient

        # T1 T2 Free Tables
        self.T1_table: Dict[int, BlockMetaData] = {}
        self.T2_table: Dict[int, BlockMetaData] = {}

        # T1 T2 priority Queue (last_accessed, -tokens, block_id, content_hash)
        self.T1_queue: List[Tuple[float,int,int,int]] = []
        self.T2_queue: List[Tuple[float,int,int,int]] = []

        # Ghost
        self.B1: OrderedDict[int, None] = OrderedDict()     # Record content_hash
        self.B2: OrderedDict[int, None] = OrderedDict()
        
        self.T1_active: set[int] = set()    # Record block_id
        self.T2_active: set[int] = set()


    def __contains__(self, block_id: int) -> bool:
        return (block_id in self.T1_table) or (block_id in self.T2_table)

    def evict(self, content_hash: int = None) -> Tuple[int, int]:
        assert content_hash not in self.T1_table and content_hash not in self.T2_table
        # Choose the Evicted Candidate
        evicted_block_id: int = -1
        evicted_content_hash: int = -1
        if content_hash in self.B1:
            # print('evict - B1 hit', content_hash)
            delta = max(1, len(self.B2) // max(1, len(self.B1)))
            self.p = min(self.p + delta, self.max_size)
            evicted_block_id, evicted_content_hash = self._replace(content_hash)
            del self.B1[content_hash]
            self.T2_active.add(evicted_block_id)
            return evicted_block_id, evicted_content_hash
        elif content_hash in self.B2:
            # print('evict - B2 hit', content_hash)
            delta = max(1, len(self.B1) // max(1, len(self.B2)))
            self.p = max(self.p - delta, 0)
            evicted_block_id, evicted_content_hash = self._replace(content_hash)
            del self.B2[content_hash]
            self.T2_active.add(evicted_block_id)
            return evicted_block_id, evicted_content_hash
        else:
            # print(f'evict miss - len T1(A), T2(A), B1, B2 | {len(self.T1_table)} ({len(self.T1_active)}) {len(self.T2_table)} ({len(self.T2_active)})  | {len(self.B1)}  {len(self.B2)}   | p: {self.p} content_hash: {content_hash}')
            L1_size = len(self.T1_table) + len(self.T1_active) + len(self.B1)
            if L1_size == self.max_size:
                if len(self.T1_table) + len(self.T1_active) < self.max_size:
                    self.B1.popitem(last=False)
                    evicted_block_id, evicted_content_hash = self._replace(content_hash)
                else:
                    assert self.T1_table
                    evicted_block_id, evicted_content_hash = self._evict_from_T1()
            elif L1_size < self.max_size:
                total_size = len(self.T1_table) + len(self.T2_table) + len(self.T1_active) + len(self.T2_active) + len(self.B1) + len(self.B2)
                # assert total_size >= self.max_size  # Can't assert it when last time to fill the cache [miss uncalled| miss called], uncalled part is missing in total_size
                if total_size >= self.max_size:
                    if total_size == 2 * self.max_size and self.B2:
                        self.B2.popitem(last=False)
                evicted_block_id, evicted_content_hash = self._replace(content_hash)    # Different, must call _replace rather than conditional
            else:
                # This case happen only when cache is not full
                assert False
                assert len(self.T1_table) + len(self.T1_active) + len(self.T2_table) + len(self.T2_active) < self.max_size
                evicted_block_id, evicted_content_hash = self._replace(content_hash)

            self.T1_active.add(evicted_block_id)
            # print('evict - evicted_content_hash', evicted_content_hash)
            return evicted_block_id, evicted_content_hash
        
        
    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int, 
            last_accessed: float, prev_block_content_hash: Optional[int] = None):
        # print(f'add - len T1, T2, B1, B2 | {len(self.T1_table)} ({len(self.T1_active)}) {len(self.T2_table)} ({len(self.T2_active)})  | {len(self.B1)} {len(self.B2)}  | p: {self.p} content_hash: {content_hash}')
        # if len(self.T1_active) < 6 and len(self.T1_active) > 0:
            # print("add - T1_active", self.T1_active)
        # assert block_id not in self.T1_free_table and block_id not in self.T2_free_table
        meta = BlockMetaData(content_hash, num_hashed_tokens, last_accessed)
    
        if block_id in self.T1_active:
            self.T1_active.remove(block_id)
            self._add_to_T1(block_id, meta)
        elif block_id in self.T2_active:
            self.T2_active.remove(block_id)
            self._add_to_T2(block_id, meta)
        else:
            assert len(self.T1_table) + len(self.T1_active) + len(self.T2_table) + len(self.T2_active) < self.max_size
            assert block_id not in self.B1 and block_id not in self.B2
            self._add_to_T1(block_id, meta)
            self._prune_ghosts()
        
        
    def update(self, block_id: int, last_accessed: float):
        assert False
        # Update last_accessed for an already-tracked block
        if block_id in self.T1_table:
            self.T1_table[block_id].last_accessed = last_accessed
        elif block_id in self.T2_table:
            self.T2_table[block_id].last_accessed = last_accessed
        else:
            raise ValueError(f"Attempting to update non-tracked block {block_id}")

    def remove(self, block_id: int):
        if block_id in self.T1_table:
            # print('remove - hit T1')
            self.T1_table.pop(block_id)
            self.T2_active.add(block_id)
        elif block_id in self.T2_table:
            # print('remove - hit T2')
            self.T2_table.pop(block_id)
            self.T2_active.add(block_id)
        else:
            raise ValueError(f"Attempting to remove non-tracked block {block_id}")

    @property
    def num_blocks(self) -> int:
        return len(self.T1_table) + len(self.T2_table)
    
    def _replace(self, content_hash_replace_for: int):
        Len_cached_T1 = len(self.T1_table) + len(self.T1_active)
        if self.T1_table and (
            (Len_cached_T1 > self.p) or
            (content_hash_replace_for in self.B2 and Len_cached_T1 == self.p)
        ):
            block_id, content_hash = self._evict_from_T1()
            self.B1[content_hash] = None
            return block_id, content_hash
        elif self.T2_table:
            block_id, content_hash = self._evict_from_T2()
            self.B2[content_hash] = None
            return block_id, content_hash
        elif self.T1_table:     # edge case: only T1 (p=c), [hit, miss], T2 only has active items
            block_id, content_hash = self._evict_from_T1()
            self.B1[content_hash] = None
            return block_id, content_hash
        raise ValueError("No usable block to replace from T1 or T2")

    def _evict_from_T1(self) -> Tuple[int, int]:
        while self.T1_queue:
            last_accessed, neg_tokens, block_id, content_hash = heapq.heappop(self.T1_queue)
            if (block_id in self.T1_table and
                    self.T1_table[block_id].last_accessed == last_accessed):
                self.T1_table.pop(block_id)
                return block_id, content_hash
        raise ValueError("No usable block left in T1 to evict")

    def _evict_from_T2(self) -> Tuple[int, int]:
        while self.T2_queue:
            last_accessed, neg_tokens, block_id, content_hash = heapq.heappop(self.T2_queue)
            if (block_id in self.T2_table and
                    self.T2_table[block_id].last_accessed == last_accessed):
                self.T2_table.pop(block_id)
                return block_id, content_hash
        raise ValueError("No usable block left in T2 to evict")
    
    def _add_to_T1(self, block_id: int, meta: BlockMetaData):
        self.T1_table[block_id] = meta
        heapq.heappush(self.T1_queue, (meta.last_accessed, -meta.num_hashed_tokens, block_id, meta.content_hash))
        self._cleanup_if_necessary(self.T1_queue, self.T1_table)

    def _add_to_T2(self, block_id: int, meta: BlockMetaData):
        self.T2_table[block_id] = meta
        heapq.heappush(self.T2_queue, (meta.last_accessed, -meta.num_hashed_tokens, block_id, meta.content_hash))
        self._cleanup_if_necessary(self.T2_queue, self.T2_table)
    
    def _prune_ghosts(self):
        # make sure the size of L1(T1 + B1) and L2 don't exceed max_size
        # assert len(self.B1) <= self.max_size and len(self.B2) <= self.max_size
        while len(self.T1_table) + len(self.T1_active) + len(self.B1) > self.max_size:
            self.B1.popitem(last=False)
        while len(self.T2_table) + len(self.T2_active) + len(self.B2) > self.max_size:
            self.B2.popitem(last=False)


    def _cleanup_if_necessary(self, queue: List[Tuple[float, int, int, int]], table: Dict[int, BlockMetaData]):
        if len(queue) > self.CLEANUP_THRESHOLD * len(table):
            self._cleanup(queue, table)

    def _cleanup(self, queue: List[Tuple[float, int, int, int]], table: Dict[int, BlockMetaData]):
        new_queue = []
        for block_id, meta in table.items():
            new_queue.append((meta.last_accessed, -meta.num_hashed_tokens, block_id, meta.content_hash))
        heapq.heapify(new_queue)

        if queue is self.T1_queue:
            self.T1_queue = new_queue
        elif queue is self.T2_queue:
            self.T2_queue = new_queue
        else:
            assert False


class CustomizedLRUEvictor(Evictor):

    CLEANUP_THRESHOLD = 50

    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}      # block_id, actual blocks in evictor
        self.active_block: Set[int] = set()     # block_id, free_table + active_block = Full Cache
        self.tail_table: TailTable = TailTable()    # key: content_hash, ordered by (last_accessed, ...)
        self.next_block: Dict[int, Set[int]] = {}    # key: content_hash

        self.eviction_window_size = 5

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self, content_hash: int = None) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        evicted_block_id: int = -1
        evicted_content_hash: int = -1
        max_num_hashed_tokens: int = -1
        eviction_window: Dict[int, Tuple[float, int, int, int]] = {}    # key: content_hash
        while len(eviction_window) < self.eviction_window_size and len(self.tail_table) > 0:
            last_accessed, num_hashed_tokens, block_id, content_hash = self.tail_table.pop_last()
            if (block_id in self.free_table and     # ensure it's not in active_block
                    self.free_table[block_id].last_accessed == last_accessed):
                eviction_window[content_hash] = (last_accessed, num_hashed_tokens, block_id, content_hash)
                if not eviction_window or num_hashed_tokens > max_num_hashed_tokens:
                    evicted_block_id = block_id
                    evicted_content_hash = content_hash
                    max_num_hashed_tokens = num_hashed_tokens

        if not eviction_window:
            raise ValueError("No usable cache memory left")
        
        # Put unchosen block back to tail_pq
        del eviction_window[evicted_content_hash]
        for _, (_, _, block_id, _) in eviction_window.items():
            self.tail_table.add(block_id, self.free_table[block_id])

        # Evict finalist
        evicted_block = self.free_table.pop(evicted_block_id)
        prev_block_content_hash = evicted_block.prev_block_content_hash
        assert prev_block_content_hash is not None
        assert evicted_content_hash in self.next_block[prev_block_content_hash]
        self.next_block[prev_block_content_hash].remove(evicted_content_hash)
        if evicted_content_hash in self.tail_table:
            self.tail_table.update(evicted_content_hash, block_id, self.free_table[block_id])   # ATTN: wrong

        return evicted_block_id, evicted_content_hash



    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, prev_block_content_hash: Optional[int] = None):
        assert prev_block_content_hash is not None
        assert block_id not in self.free_table and block_id in self.active_block
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed,
                                                  prev_block_content_hash)
        self.active_block.remove(block_id)
        
        if block_id in self.active_block:
            pass    # do nothing
        else:
            self.next_block[prev_block_content_hash].add(content_hash)
            if prev_block_content_hash in self.tail_table:
                self.tail_table.update(prev_block_content_hash, block_id, self.free_table[block_id])
            else:
                self.tail_table.add(block_id, self.free_table[block_id])

        # self._cleanup_if_necessary()

    def update(self, block_id: int, last_accessed: float):
        assert False
        self.free_table[block_id].last_accessed = last_accessed

    def _cleanup_if_necessary(self):
        if len(self.priority_queue) > CustomizedLRUEvictor.CLEANUP_THRESHOLD * len(
                self.free_table):
            self._cleanup()

    def _cleanup(self):
        new_priority_queue: List[Tuple[float, int, int, int]] = []

        for block_id, block in self.free_table.items():
            new_priority_queue.append(
                (block.last_accessed, -block.num_hashed_tokens, block_id,
                 block.content_hash))
        heapq.heapify(new_priority_queue)

        self.priority_queue = new_priority_queue

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self.free_table.pop(block_id)
        self.active_block.add(block_id)

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)