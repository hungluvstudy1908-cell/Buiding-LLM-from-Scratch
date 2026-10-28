gpt-from-scratch/
├── README.md                 # Mục tiêu, cách chạy, nhật ký tiến độ
├── pyproject.toml            # (hoặc requirements.txt)
├── .gitignore                # Bỏ data/, checkpoints/, .venv/
│
├── configs/
│   ├── tiny.yaml             # Model nhỏ để debug (~1M params)
│   └── small.yaml            # Model lớn hơn để train thật
│
├── data/
│   ├── raw/                  # Text gốc (ví dụ: the-verdict.txt)
│   └── processed/            # Token ids đã mã hoá (.bin / .pt)
│
├── src/gpt/
│   ├── __init__.py
│   ├── tokenizer.py          # Simple tokenizer → BPE
│   ├── dataset.py            # Sliding window, DataLoader
│   ├── attention.py          # Self-attention → causal → multi-head
│   ├── layers.py             # LayerNorm, GELU, FeedForward
│   ├── model.py              # TransformerBlock + GPTModel
│   ├── train.py              # Training loop, eval, checkpoint
│   ├── generate.py           # Greedy, temperature, top-k
│   └── utils.py              # Seed, device, logging
│
├── tests/
│   ├── test_tokenizer.py
│   ├── test_attention.py     # Kiểm tra shape + causal mask
│   └── test_model.py         # Forward pass, overfit 1 batch
│
├── scripts/
│   ├── prepare_data.py
│   ├── run_train.py
│   └── run_generate.py
│
├── notebooks/                # Thử nghiệm, vẽ loss curve
├── checkpoints/              # Model đã lưu (không commit)
└── logs/