# Eviction Strategy

## 🔧 Modified Files & Change Summary

### `prefix_caching_block.py`
- Modify interface for ARC to pass the content_hash.
- Print the content_hash for tracing.

### `Evictor.py`
- Link the eviction strategies.

### `customized_evictor.py`
- Implemented DBL, ARC eviction strategy.

### `Common.py`
- Return completed block hit rate throughout all requests rather than a recent history.
- Potentially print hit rate & hit count for debug. 

### `llm_engine.py`
- Add printer for hit rate.

## 🐞 Hardcoding Notice
- Manually choose the eviction strategy in `make_evictor()` in `Evictor.py`.
- Content_hash trace is logged defaultly at: `/home/shenyang/tests/result/block_log.txt`