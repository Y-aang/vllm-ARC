import heapq
from typing import Dict, List, Tuple

from vllm.core.evictor import BlockMetaData

Entry = Tuple[float, int, int, int]
# (last_accessed, -num_hashed_tokens, block_id, content_hash)


class TailTable:
    # A ordered table to record all tail blocks.
    # # Support: ordered by (last_accessed, ...), add, pop, update.
    CLEANUP_THRESHOLD = 50

    __slots__ = ("_heap", "_table")

    def __init__(self):
        self._heap: List[Entry] = []
        self._table: Dict[int, Entry] = {}  # content_hash -> all tail entry

    @staticmethod
    def _make_entry(block_id: int, meta: BlockMetaData) -> Entry:
        return (
            meta.last_accessed,
            -meta.num_hashed_tokens,
            block_id,
            meta.content_hash,
        )

    def __contains__(self, content_hash: int) -> bool:
        # Suport: if block_id in tail_table
        return content_hash in self._table

    def add(self, block_id: int, meta: BlockMetaData) -> None:
        entry = self._make_entry(block_id, meta)
        self._table[meta.content_hash] = entry
        heapq.heappush(self._heap, entry)
        self._cleanup_if_necessary()

    def pop_last(self) -> Entry:
        while self._heap:
            entry = heapq.heappop(self._heap)
            last_accessed, num_hashed_tokens, block_id, content_hash = entry

            # verify the entry is still valid only when it matches the item in self._table
            if content_hash in self._table and self._table[content_hash] == entry:
                del self._table[content_hash]
                return last_accessed, -num_hashed_tokens, block_id, content_hash

        raise ValueError("TailTable is empty")

    def update(self, old_content_hash: int, new_block_id: int, meta: BlockMetaData) -> None:
        # lazy delete the old entry by removing the item from self._table
        self._table.pop(old_content_hash, None)
        # add the new entry
        self.add(new_block_id, meta)
        self._cleanup_if_necessary()

    def _cleanup_if_necessary(self) -> None:
        live = len(self._table)
        if live == 0:
            self._heap.clear()
            return

        if len(self._heap) > TailTable.CLEANUP_THRESHOLD * live:
            self._cleanup()

    def _cleanup(self) -> None:
        new_heap = list(self._table.values())
        heapq.heapify(new_heap)
        self._heap = new_heap

    def __len__(self) -> int:
        return len(self._table)

