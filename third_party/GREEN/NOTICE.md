# Vendored GREEN (modified)

Source: https://github.com/Stanford-AIMI/GREEN, commit
`d72c1951e69683b93405968901fd4f14cc6c27b0`, MIT License (see `setup.py`).

Only used by the optional GREEN reward server (`GRPO/reward_server_green_v3.py`).
The upstream package does **not** work with that server; install this copy instead:

```bash
pip install -e third_party/GREEN
```

Changes relative to upstream (full diff in `ctrg_modifications.patch`):

- `GreenSingle`: new single-GPU subclass of `GREEN` without `torch.distributed`
  initialisation; exported from `green_score/__init__.py`.
- `GREEN.__call__` / `infer` return the per-sample GREEN scores directly (summary
  statistics are skipped).
- Generation uses `max_new_tokens=512` instead of `max_length=2048`, which raises
  an error on transformers >= 4.40 for long report pairs.
- Batch size 8 -> 2; `requirements.txt` pins `datasets==3.1.0` and drops
  training-only dependencies.
