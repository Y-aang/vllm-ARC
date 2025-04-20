import heapq
from typing import Dict, List, Tuple
from vllm.core.evictor import BlockMetaData
from vllm.core.evictor import Evictor
from collections import OrderedDict, deque

class CustomizedEvictor(Evictor):

    def __init__(self, max_size:int=134, k:int=50):
        self.max_size = max_size
        self.k = k
        self.A1in: OrderedDict[int, BlockMetaData] = OrderedDict()
        self.Am: OrderedDict[int, BlockMetaData] = OrderedDict()
        self.A1out: Deque[int] = deque(maxlen=self.k)

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.A1in or block_id in self.Am

    def evict(self) -> Tuple[int, int]:
        print('[EVICT] len(self.A1in, A1out, Am):', len(self.A1in), len(self.A1out), len(self.Am))
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
        print('[ADD] len(self.A1in, A1out, Am):', len(self.A1in), len(self.A1out), len(self.Am))
        meta = BlockMetaData(content_hash, num_hashed_tokens, last_accessed)

        if block_id in self.Am:     # Already in Am → refresh LRU position
            self.Am.move_to_end(block_id)
            return
        if block_id in self.A1in:   # Already in A1in → no-op
            return
        if block_id in self.A1out:  # Ghost hit → promote to Am
            self.A1out.remove(block_id)
            self._add_to_Am(block_id, meta)
        else:                       # First time → insert into A1in
            self.A1in[block_id] = meta

    def update(self, block_id: int, last_accessed: float):
        if block_id in self.A1in:
            self.A1in[block_id].last_accessed = last_accessed
        elif block_id in self.Am:
            self.Am[block_id].last_accessed = last_accessed
            self.Am.move_to_end(block_id)

    def remove(self, block_id: int):
        if block_id in self.A1in:
            self.A1in.pop(block_id)
        elif block_id in self.Am:
            self.Am.pop(block_id)
        else:
            raise ValueError(f"Block {block_id} not tracked")
        

    @property
    def num_blocks(self) -> int:
        return len(self.A1in) + len(self.Am)
    
    def _add_to_Am(self, block_id: int, meta: BlockMetaData):
        self.Am[block_id] = meta
        self.Am.move_to_end(block_id)
