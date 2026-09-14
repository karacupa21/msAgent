# ExGraph P2.0 testing

Single zip: `exgraph-p1.1.zip`. See `APPLY.md`.
Do not layer older zips on top of this one.

```
python -m pytest tests/ut/exgraph \
  tests/ut/skill_evolver/test_exgraph_context.py \
  tests/it/exgraph/test_evolver_value.py -q --noconftest
```

## Fixes in this rewrite

1. **Writable side-effect paths.** `test_write_value_report` and
   `test_growth_html_highlights_new_over_trajectories` must not mkdir
   `/home/workdir/artifacts` (that path exists only on the authoring
   machine). Order: `EXGRAPH_VALUE_REPORT` / `EXGRAPH_GROWTH_HTML` →
   `<repo>/artifacts/` only if that directory already exists and is
   writable → pytest `tmp_path`.
2. **Agent-scoped recipes.** `rebuild_overlay` runs `mine_cross_session`
   per agent. Accuracy `read_file>grep` must not `INSTANTIATE` a
   Profiler recipe. Covered by
   `test_overlay_does_not_instantiate_other_agents` and the growth test
   (`case:run-a1` not in INSTANTIATES sources).

## Intensive A/B

`test_evolver_value.py` appendix **suffix** only:

- `SkillDoc` when a proposal is planted;
- `fixed_by` and `outcome=`;
- no `Episodes:`, no `### Episode`, no `Evidence:`.

Growth HTML uses live `remember_thread` unless `EXGRAPH_VIZ_DEMO=1`.

Optional:

```
EXGRAPH_VALUE_REPORT=$PWD/artifacts/exgraph_evolver_value_report.md \
EXGRAPH_GROWTH_HTML=$PWD/artifacts/exgraph_growth_demo.html \
  python -m pytest tests/it/exgraph/test_evolver_value.py -q --noconftest
```

Create `artifacts/` first if you want those files next to the repo.

P2.0 (not in this zip): `/skill-mine` with graph on vs
`MSAGENT_EXGRAPH_DISABLED=1` on a real project, score the written
`SKILL.md`.

Default config is off. Intensive tests set `MSAGENT_EXGRAPH_ENABLED=1`.

P2.0 intensive appendix must include Similar cases and Insights when an accepted skill is planted.
