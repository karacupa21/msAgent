# Experience Graph — Architecture

Status: P0–P2.3 + **P2.2** classify A/B harness, schema version 3.
Branch: `feature/experience-graph`.

## 1. Purpose

Turn recorded msAgent trajectories into an **experience graph**: cases,
outcomes, tool paths, corrections, recipes, and pointers at skill files
distilled from those threads.

Not an entity KG. Atomic unit is a **case** (one user turn).

Primary consumers (all current, not “P3”):

- offline inspection (`python -m msagent.exgraph.export`);
- growth picture (`python -m msagent.exgraph.visualize`);
- Skill Evolver classify appendix (`attach_stored_graph`).
  Exgraph does not write `SKILL.md`.

## 2. Task grain

One `TaskAnchor` per thread (`task:thread:{thread_id}`). Each `Case`
stores that turn’s user text as `x` so anchors can split later without
renaming cases.

## 3. Schema

Nodes: `Thread`, `TaskAnchor`, `Case`, `Step`, `SubagentRun`, `SkillDoc`,
`Episode`, workspace `Recipe`, workspace `Insight`.

Edges: `HAS_TASK`, `CONTAINS`, `NEXT_CASE`, `HAS_STEP`, `PARENT_OF`,
`DELEGATES`, `IN_SUBAGENT`, `DERIVED_SKILL`, `HAS_EPISODE`, `FIXED_BY`,
`INSTANTIATES`, `SIMILAR_TO`, `INSIGHT_OF`, `INSIGHT_SKILL`.

Case payload: `x`, `y`, `r` (`golden` | `warning` | `unknown`), `sigma`
(tool path, errors, retries, approvals, tokens).

Outcome policy v1 is unchanged: `turn.end=completed` is not success;
`golden` only via `--outcome success` on a non-warning turn.

## 4. Package (complete in this zip)

```
src/msagent/exgraph/
    __init__.py     no imports
    config.py       config.exgraph.yml + live kill switch
    schema.py       ids, Node, Edge, CaseRecord
    sources.py      only import of trajectory_recorder
    cases.py        L0 ingest
    skills.py       SkillDoc path scan
    enrich.py       extract_episodes + FIXED_BY
    workspace.py    mine_cross_session overlay + similar + insight
    similar.py      SIMILAR_TO (Jaccard, same agent)
    insight.py      Insight from accepted SkillDoc + recipe
    consumer.py     classify appendix (relations only)
    store.py        JSONL shards under <state>/exgraph/
    export.py       build | show | export | viz
    visualize.py    growth HTML
src/msagent/skill_evolver/exgraph_context.py   fail-open hook
```

This zip does **not** vendor `mining.py` / `direct_skill_generation.py`.

Storage:

```
<project-state>/exgraph/<agent>_<thread_id>/
<project-state>/exgraph/_workspace/          # recipes only
```

## 5. Tenancy

Store follows project state — same isolation as trajectories. Several
users on one install get several graphs if they use several working
dirs. Sharing a project folder shares the graph. There is no
process-wide singleton and no company-wide dump of raw cases.

The trajectory recorder's `output.scope: shared` (one trajectory store
for every workspace) does not change this: graphs stay under the project
state, and `load_source_trajectory` / `build --all` read only the
threads recorded in their `--working-dir` (default: cwd).

A later optional `MSAGENT_EXGRAPH_ROOT` may point at a **read-mostly
overlay** of recipes + accepted SkillDocs. That is not P1.1.

## 6. Skill Evolver contract

Detectors stay in `skill_evolver.features`. Exgraph only materializes
them. `select_trajectories` / `CROSS_SESSION_LIMIT=20` stay evolver-owned.

After the evolver builds its episode bundle, `pipeline.run_thread`
calls `attach_stored_graph` (the only hook). Fail-open. `valid_seq`
remains the evolver episode seqs. Mode is **not** on
`config.skill.evolver.yml` (`extra: forbid`).

| `evidence_mode` | Classify string |
|---|---|
| `episodes` | original bundle; no shard |
| `hybrid` | bundle + relation appendix (default when the graph is on) |
| `graph` | relations first, then `## Episode bundle (supporting, citable)` |

`MSAGENT_EXGRAPH_EVIDENCE_MODE` overrides YAML. Disabled / kill switch
forces `episodes`.

P1.1 appendix contains **only relations**:

- cases with `outcome`, tool path, first `sigma` error;
- `FIXED_BY`;
- SkillDoc name / status / path;
- recipes this thread instantiates (`ngram`, support, other thread ids).

P2.0 also lists `Similar cases` (same-agent tool+text Jaccard) and
`Insights` (accepted SkillDoc + recipe support≥2). No invented prose.

It does **not** repeat episode kinds. Cap 1600 characters. No `Evidence:`
seqs.

## 7. Default off (P1.2 merge)

Shipped YAML and the pydantic default are `enabled: false`. A merge into
`extract-session-history-generate-skill-md-and-other` does not change
Skill Evolver unless someone opts in.

- `MSAGENT_EXGRAPH_ENABLED=1` turns the graph on for that process.
- `MSAGENT_EXGRAPH_DISABLED=1` (`true`/`yes`/`on`) always wins.
- Exgraph unit/intensive tests opt in via fixture; other tests do not.

Checked live on CLI `build`, `remember_thread`, the evolver hook, and the
appendix. Inspection `show`/`export` of an existing shard still works.

P2.1b: after a thread writes at least one proposal, `pipeline` asks
`fill_overlay_insights` to set `Insight.text` with the same
`CountingLlm` (limit still `max_llm_calls_per_thread`). Overlay rebuild
stays deterministic. Empty overlay / exhausted budget / kill switch /
`insight.fill_llm: false` → structural Insight only. Daemon is a third
`run_thread` caller; it inherits the hook. FEATURES_VERSION is 5
(evolver-owned).

P2.2: `python -m msagent.exgraph.export ab` compares classify *input*
under evidence_mode values. No LLM, no new node types. Full proposal
A/B is `/skill-mine` twice with `MSAGENT_EXGRAPH_EVIDENCE_MODE`.

## 8. Later

BM25 text similar via `skill_evolver.retrieval.tokenize` (P2.3), live
`/skill-mine` A/B (P2.2), argument-aware `mine_cross_session`, outcome
`recovered`, `finish_turn` enqueue, dense+RRF, `/exgraph` slash command.

## 9. CLI

```
python -m msagent.exgraph.export build --thread <id> [--outcome success|fail]
python -m msagent.exgraph.export build --all
python -m msagent.exgraph.export show --thread <id>
python -m msagent.exgraph.export export --thread <id> --format json
python -m msagent.exgraph.export viz -o artifacts/exgraph_growth_demo.html
python -m msagent.exgraph.visualize --fixtures tests/fixtures/trajectories -o artifacts/exgraph_growth_demo.html
```

`EXGRAPH_VIZ_DEMO=1` or `--demo` uses canned snapshots. Default viz
ingests live fixtures.
