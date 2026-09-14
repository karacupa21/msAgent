# P2.4 — freeze and harden

GitHub tip when this was cut: `ea697b2` (P2.3 BM25 already on the branch).
P2.2 `ab` was not committed; this zip lands it plus harden.

## Frozen (do not add in this slice)

No new node/edge types. No embeddings. No `mine_cross_session` fork.
No `finish_turn` live enqueue. No `/exgraph` slash command.

## What this zip adds

- `export ab` classify-input A/B (P2.2, cleaned)
- `--report-a/--report-b` diff of two `/skill-mine` decision JSON files
- kill-switch respected when `force_enable=False`
- `test_surface.py` so copying only tests without `insight.py` fails fast
- sidecar `.json` next to the markdown A/B report

## After apply

```bash
uv run pytest tests/ut/exgraph/test_ab.py tests/ut/exgraph/test_surface.py -q --noconftest
```
