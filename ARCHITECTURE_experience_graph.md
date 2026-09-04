# Experience Graph — Architecture

Status: P0 implemented (`src/msagent/exgraph/`), schema version 1.
Branch: `feature/experience-graph`.

## 1. Purpose

Turn recorded msAgent trajectories into an **experience graph**: a relational
store of what the agent tried, whether the attempt failed, which tools ran,
and (when present) which skill proposal was distilled from that thread.

This is not a knowledge graph of world entities. Atomic units are **cases**
(one user turn / attempt), not triples like `Service → dependsOn → Redis`.

Primary consumers:

- offline inspection (`python -m msagent.exgraph.export`);
- later recipe / insight mining (P1+);
- Skill Evolver evidence (P3), without writing `SKILL.md` itself.

## 2. Task grain (P0 decision)

Interactive msAgent work is a multi-turn investigation. Hashing every user
message into its own task would split follow-up turns and break `FIXED_BY` in P1.

P0 therefore uses **one `TaskAnchor` per thread** (`task:thread:{thread_id}`).
Each `Case` still stores that turn's user text as `x`, so a later pass can
split anchors without rebuilding identities of cases or steps.

A `Thread` node sits above the anchor for provenance (`working_dir`, model).

## 3. P0 schema

Nodes: `Thread`, `TaskAnchor`, `Case`, `Step` (`kind=tool|llm`),
`SubagentRun`, optional `SkillDoc`.

Edges: `HAS_TASK`, `CONTAINS`, `NEXT_CASE`, `HAS_STEP`, `PARENT_OF`,
`DELEGATES`, `IN_SUBAGENT`, `DERIVED_SKILL`.

Case payload (EXG tuple on recorder events):

- `x` — user message of the turn
- `y` — last root-agent assistant text
- `r` — `golden` | `warning` | `unknown`
- `sigma` — tool path, errors, retries, approvals, tokens, subagents, skills

Prelude turns (`run_id=__prelude__`) are skipped.

## 4. Outcome policy v1

`turn.end status=completed` is not success.

- `warning` if the turn ended in `error` or any tool span has `status=error`
- `golden` only when the caller passes `--outcome success` on a non-warning turn
- `unknown` otherwise
- `--outcome fail` forces non-warning turns to `warning`

The policy name is stored on every case (`outcome_policy`) so labels can be
recomputed later.

## 5. Package

```
src/msagent/exgraph/
    __init__.py     no imports (lightweight CLI)
    config.py       config.exgraph.yml loader
    schema.py       ids, Node, Edge, CaseRecord
    sources.py      the only import of trajectory_recorder
    cases.py        deterministic ingest from the typed Trajectory model
    skills.py       optional SkillDoc scan
    store.py        upsert JSONL under <state>/exgraph/
    export.py       CLI build | show | export
```

Storage (derived, never overwrites source JSONL):

```
<project-state>/exgraph/<agent>_<thread_id>/
    manifest.json
    nodes.jsonl
    edges.jsonl
    cases.jsonl
```

Rebuild is upsert-by-id.

## 6. Skills from this trajectory

When `skills.enabled` is true, ingest looks at:

- `<working_dir>/.proposals/<thread>/**/SKILL.md` (Skill Evolver drafts)
- `<working_dir>/skills/**/SKILL.md` whose footer or `provenance.json`
  lists this `thread_id`

Those files become `SkillDoc` nodes linked with `DERIVED_SKILL`. Missing
folders are normal and silent.

## 7. What P0 does not do

Recipes, `SIMILAR_TO`, `FIXED_BY`, LLM insights, slash command `/exgraph`,
live retrieval, embeddings, external graph databases.

Documented future hook for online growth: `trajectory_hooks.finish_turn` may
enqueue `exgraph` ingest behind `online: false`. It must not block the agent.

## 8. CLI

```
python -m msagent.exgraph.export build --thread <id> [--outcome success|fail]
python -m msagent.exgraph.export build --path /path/to/thread.jsonl
python -m msagent.exgraph.export show --thread <id>
python -m msagent.exgraph.export export --thread <id> --format json
```

Kill switch: `MSAGENT_EXGRAPH_DISABLED=1`.
