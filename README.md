# Buiding-LLM-from-Scratch
A GPT language model built from scratch in PyTorch: tokenizer, attention, Transformer blocks, training loop and text generation, with tests for every module

# GPT from Scratch

A small GPT language model, implemented from the ground up in PyTorch
without relying on high-level libraries such as `transformers`.

Hình dung: GPT giống một chiếc máy đoán từ tiếp theo. Tokenizer cắt câu
thành mảnh, attention cho mỗi mảnh "nhìn lại" các mảnh trước nó, và quá
trình training dạy máy đoán mảnh kế tiếp ngày càng chính xác.

## Highlights
- Every component is written by hand: tokenizer, causal multi-head
  attention, LayerNorm, GELU, feed-forward, Transformer block
- Unit tests for tensor shapes, the causal mask, and overfitting one batch
- Two configs: `tiny` (~1M params, for debugging) and `small` (real training)
- Generation with greedy decoding, temperature and top-k sampling

## Status
- [x] `attention.py`: causal multi-head self-attention
- [ ] tokenizer, dataset
- [ ] layers, model
- [ ] training loop, generation
