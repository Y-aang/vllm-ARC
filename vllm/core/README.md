# Eviction Strategy

## 🔧 Modified Files & Change Summary

### [`prefix_caching_block.py`](./block/prefix_caching_block.py)
- Modify interface for ARC to pass the content_hash etc.
- Print and log content_hash for tracing.

### [`evictor.py`](./evictor.py)
- Route the eviction strategies.

### [`customized_evictor.py`](./customized_evictor.py)
- Implemented DBL, ARC eviction strategy.

### [`common.py`](./block/common.py)
- Return the KV block hit rate throughout all requests rather than the one from a recent history.
- Potentially print hit rate & hit count for debug. 

### [`llm_engine.py`](../engine/llm_engine.py)
- Print the hit rate.

## 📑 Environment Variables
Control where the block/content-hash trace log is written. For example:
```
export BLOCK_LOG_FILE_PATH=/tmp/block_log.txt
```
