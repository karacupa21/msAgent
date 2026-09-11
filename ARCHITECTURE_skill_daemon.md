# Skill Daemon — Architecture

Component: unattended SKILL.md mining (`msagent-skill-daemon`)
Status: working draft; reflects branch `extract-session-history-generate-skill-md-and-other`
as of 2026-09-10 — config `schema_version` 1, ledger version 1, inbox version 1.
Companion document: `ARCHITECTURE_skill_evolve.md` (the pipeline this component drives).

## 1. Purpose

Skill Evolver distils a recorded session into a `SKILL.md` proposal, but only when a user
types `/skill-mine`. Everything the pipeline needs is already on disk the moment a session
ends; nothing reads it until somebody remembers to. Meanwhile the cross-session evidence
that `features.mine_cross_session` depends on matures inside a pool of the newest 20
trajectories and then falls out of it again.

The daemon closes that gap: a scheduled batch finds the finished trajectories nobody has
mined, runs them through the **existing** pipeline, and records what it decided.

What it deliberately does **not** change is the trust boundary. Output is an inactive
proposal under `.proposals/`, exactly as section 9 of `ARCHITECTURE_skill_evolve.md`
requires; promotion stays a human act through `/skill-review`. The daemon's only new
user-facing surface is one line at CLI startup saying how many proposals are waiting.

## 2. User-facing behavior

| Invocation | Effect |
|---|---|
| `msagent-skill-daemon [--once]` | One tick: scan, mine what is due, record, exit |
| `msagent-skill-daemon --dry-run` | Select and gate threads; no LLM is created, no ledger row is written |
| `msagent-skill-daemon --watch [SECONDS]` | The same tick in a sleep loop, for a host without a timer |
| `msagent-skill-daemon --force` | Ignore `schedule.min_interval_seconds` for this run |
| `msagent-skill-daemon --project <dir\|id>` | Restrict the tick to one project |
| `msagent-skill-daemon -v` | Log to stderr at INFO; systemd collects it into the journal |

`python -m msagent.skill_evolver.daemon` is equivalent to the console script.

Exit codes: `0` the tick ran (or was disabled, busy, or too soon), `2` the configuration is
broken and is printed key by key, `1` anything unexpected. A thread that failed inside a
healthy tick is a warning, not a failed unit — the ledger row and the decision report carry
the detail.

Off by default. `enabled: true` in `~/.msagent/config/config.skill.daemon.yml`, or
`MSAGENT_SKILL_DAEMON_ENABLED=1`, opts in; `MSAGENT_SKILL_DAEMON_DISABLED=1` always wins.

## 3. Component inventory

| File | Role |
|---|---|
| `src/msagent/skill_evolver/daemon/config.py` | `config.skill.daemon.yml` schema_version 1: `SkillDaemonConfig` (strict, `extra="forbid"`), the env kill switch, `daemon_state_dir()`, `SkillDaemonConfigError` with dotted paths |
| `src/msagent/skill_evolver/daemon/ledger.py` | The sqlite record of what has been mined: `Ledger`, `Verdict`, `content_sha256`, `methodology_fingerprint`, the status vocabulary, poisoning after `max_attempts` |
| `src/msagent/skill_evolver/daemon/discovery.py` | `discover_projects`, `scan_project`, `preflight`, `incompleteness`, `iter_scans`, `PoolNotReady` — which projects exist and which trajectories are finished |
| `src/msagent/skill_evolver/daemon/lock.py` | `tick_lock`: an advisory whole-file lock so two schedulers cannot mine the same threads |
| `src/msagent/skill_evolver/daemon/runner.py` | `run_once`, `TickResult` — the tick: budget, per-agent RunContext, the loop over `SkillMiningHandler._mine_thread`, the ledger rows |
| `src/msagent/skill_evolver/daemon/inbox.py` | The per-project notification file and `print_daemon_notice`, the CLI startup banner |
| `src/msagent/skill_evolver/daemon/cli.py` | `main()`: argument parsing, stderr logging, exit codes |
| `src/msagent/skill_evolver/daemon/__main__.py` | `python -m` convenience wrapper |
| `src/msagent/skill_evolver/daemon/README.md` | Operator guide: install, schedule, where the results land, troubleshooting |
| `resources/configs/default/config.skill.daemon.yml` | Packaged default, the commented schema_version 1 document |
| `resources/systemd/msagent-skill-daemon.{service,timer}` | User units plus their install instructions |
| `tests/ut/skill_evolver/test_daemon_{config,ledger,discovery,inbox,runner}.py` | 67 tests; the runner suite drives the real pipeline on a scripted LLM |

Modified files (integration points — four lines in total):

| File | Change |
|---|---|
| `pyproject.toml` | `msagent-skill-daemon` console script |
| `src/msagent/core/constants.py` | `CONFIG_SKILL_DAEMON_FILE_NAME` |
| `src/msagent/core/storage_layout.py` | `_seed_skill_evolver_defaults()` seeds the daemon config too (copy-if-missing) |
| `src/msagent/cli/core/session.py` | One `print_daemon_notice(...)` call in `start()` |

**`mining.py` and `pipeline.py` are not modified.** The daemon is a third thin caller of
`pipeline.run_thread`, in the spirit of section 11 of the evolver architecture: a rule that
exists once cannot drift between the callers.

## 4. Runtime layout

```text
~/.msagent/                                   (MSAGENT_HOME)
├── config/
│   └── config.skill.daemon.yml               # schema_version 1 (section 5)
└── state/
    ├── skill-daemon/                         # the daemon's own state, shared by all projects
    │   ├── ledger.sqlite3                    # what has been mined (section 6)
    │   └── tick.lock                         # one tick at a time (section 8)
    ├── trajectories/                         # input, read-only — output.scope: shared only
    └── projects/<slug>-<sha12>/
        ├── trajectories/                     # input, read-only — output.scope: workspace (default)
        └── skill-evolver/
            ├── decisions/                    # the pipeline's own reports, command: "skill-daemon"
            └── daemon-inbox.json             # the CLI banner reads this (section 9)
```

Proposals land where the interactive command puts them: `<output_root>/.proposals/<thread>/<name>/`.

With the recorder's `output.scope: shared` (ARCHITECTURE_trajectory_recorder.md, section 5) every
project resolves the one `state/trajectories/` directory. Everything else stays per project: a thread
is attributed to the project its `recorder.attach.working_dir` names (section 7), so its report, inbox
entry, ledger row and proposals are the ones the workspace scope would have produced.

## 5. Configuration (schema_version 1)

A file of its own rather than a section in `config.skill.evolver.yml`: that file is
schema_version 2 with `extra="forbid"`, so a new section would force a schema bump and an
in-memory migration on every existing home. The key is `schema_version`, not `version`,
matching its sibling — these component files version their own schema and are not tied to
the application version that `tests/ut/configs/test_config_versions.py` enforces elsewhere.

| Key | Meaning |
|---|---|
| `enabled` | Off by default; an unattended run spends LLM calls |
| `schedule.quiet_period_seconds` | How long a trajectory must be idle before it counts as finished (900) |
| `schedule.min_interval_seconds` | Refuse a tick sooner than this, whatever the timer says (1800; section 11 explains why it sits below the timer period) |
| `schedule.max_threads_per_tick` | Threads mined per tick, across all projects (5) |
| `schedule.max_llm_calls_per_tick` | Hard ceiling; the tick stops **before** it would exceed this (60) |
| `scope.projects` / `project_dirs` / `agents` | `all` or `listed`; empty `agents` means every agent |
| `mining.model` | `null` resolves the model the CLI would use; an alias puts the background on a cheaper model |
| `mining.policy` / `mining.demo` | The same values as `--policy` and `--demo` |
| `mining.remine_on` | `content_change`, `methodology_change` (section 6) |
| `mining.max_attempts` | Consecutive failures of one file before it is excluded for good |
| `notify.inbox` | Write the per-project inbox for the CLI banner |

Loading is **loud**: a malformed file, an unknown key or an out-of-range value raises
`SkillDaemonConfigError` listing every problem with its dotted path, and the tick stops.
Unlike the experience-graph loader there is no degradation to defaults — a background job
running under a configuration nobody chose is worse than one that does not run.

Env: `MSAGENT_SKILL_DAEMON_CONFIG` (explicit path), `_DISABLED`, `_ENABLED`. The flags are
re-read on every call, so flipping them needs no restart; DISABLED always wins.

## 6. Idempotency: the ledger

Nothing in the interactive commands records that a thread was mined. Re-running
`/skill-mine` re-mines every selected thread, spends the budget again and writes another
proposal folder (`-2`, `-3` via `writer._reserve_dir`). A scheduled job cannot behave that
way, so the daemon keeps its own record: one sqlite file under
`<MSAGENT_HOME>/state/skill-daemon/`, keyed by `(project_id, thread_id)`.

Each row carries the file's `content_sha`, a `fingerprint` of the methodology, the outcome
`status` (`mined`, `gate_skip`, `nothing`, `rejected`, `failed`, `poisoned`), the consecutive
failure count, the proposals written and the LLM calls actually spent.

A candidate is mined when:

- there is no row; or
- **`content_change`** — the file hash moved: the user continued the thread, so evidence
  exists that the recorded verdict never saw; or
- **`methodology_change`** — the fingerprint moved. The fingerprint hashes `FEATURES_VERSION`,
  the sha256 of all three stage prompts and `EffectiveRules.as_record()`, and is computed
  **before** any LLM exists. New prompts or a new policy mean the old verdict no longer
  describes what a run would decide today.

Both reasons are configurable through `mining.remine_on`, so a user who wants strictly
one-shot mining can switch either off.

A `failed` row is due again on the next tick even when nothing changed: a failure is often
transient (a model endpoint that was down), and `record()` counts the consecutive failures
of the same content, so the retry cannot loop forever.

`status = poisoned` is terminal: after `max_attempts` consecutive failures on the *same
content* the row is excluded from every future tick, with one ERROR naming the file.
Failures count per content hash, so a rewritten thread gets a fresh chance.

## 7. Deciding a trajectory is finished

**There is no `session.end` event.** The recorder opens the file lazily, appends a line and
closes it again — no sentinel, no rename, no finalizer, no `atexit`
(`ARCHITECTURE_trajectory_recorder.md`). Adding one would edit the SCHEMA_VERSION 1 write
path, which the project rules forbid. Completeness is therefore inferred:

```text
candidate = quiescent AND complete AND in scope AND due
```

- **quiescent** — `now - st_mtime >= quiet_period_seconds`. The same signal `/skill-mine
  --since` already trusts, and for the same reason: mtime always exists and means last
  activity, while `started_at` is the first event's timestamp and can be empty.
- **complete** — the reader's own markers: the last turn is not `truncated` (a Ctrl+C
  session leaves no `turn.end`) and no `ToolCall` is still `orphan`. A thread stopped by
  `recorder.limit` counts as complete: nothing more will ever be appended, so waiting would
  only let it go stale.
- **in scope** — `scope.agents` and `scope.projects`.
- **due** — the ledger verdict of section 6.

Projects are enumerated from `<home>/state/projects/*/project.json`, whose `working_dir` is
what the pipeline needs to resolve skills and the output root. Candidates are grouped by
**(project, agent)**: both the skill catalogue and the cross-session pool are agent-scoped.

**Shared store.** Under the recorder's `output.scope: shared` every project resolves the same
directory, and scanning it per project would parse each file once per project and mine each thread
once per project. `discovery.iter_shared_scans` reads the store once per tick instead: `preflight`
over the whole store (one empty file defers the whole tick, section 8), then each file goes to the
project whose working directory its first event (`recorder.attach.working_dir`) names, compared by
`reader.workspace_key`. A file whose origin is unknown, relative, or not among this tick's projects
(out of `scope.projects`, not selected by `--project`, no `project.json` any more) is logged and left
alone; every project still gets a `ScanResult`, so its inbox is stamped. The per-file rules are the
same `_examine` as `scan_project`. The cross-session pool is narrowed to the origin workspace
(`_pool_and_targets(..., workspace=)`), the pool `/skill-mine` builds in that workspace, so a
background verdict equals an interactive one. Each tick reloads the recorder config
(`load_trajectory_config(force_reload=True)`), so a `--watch` process sees a scope change without a
restart.

## 8. Safety of reading files another process is writing

Three distinct hazards, with three different answers:

1. **A torn last line** is a non-issue. `iter_numbered_events` skips unparsable lines while
   keeping their physical numbers, so an `EvidenceRef` written before the tear stays valid;
   appending never moves an earlier line. Reading a growing file is an explicitly supported
   pattern of the reader.
2. **An empty `.jsonl`** is fatal to the *whole project*, and that is the surprising one.
   `load_trajectories` filters on each file's first event through `_peek_header`, which
   raises `TrajectoryReadError` for a file with no events — and it is called for **every**
   `*.jsonl` in the directory. A session that has just created its file but not yet written
   to it would kill the selection for every thread of that project. `discovery.preflight`
   therefore checks each file for a first event before anything else and raises
   `PoolNotReady`; `iter_scans` defers that project to the next tick with a warning. The
   condition is transient and heals itself.
3. **A quiescent file that still will not parse** is a genuine defect, not a race, and it
   is handled in two places because it can appear in two roles. As a *candidate* it is
   logged at ERROR and skipped by `scan_project`; if the pipeline later fails on it, the
   ledger counts it against `max_attempts` and eventually poisons it. As a member of the
   *cross-session pool* it is worse: `load_trajectories` refuses the whole listing, so one
   bad neighbour would take every thread of that agent with it. `/skill-mine` ends the
   command there, which is right for a user watching the output; a timer must not lose
   every other project to it, so `_run_project` catches `TrajectoryReadError` per agent,
   counts a failure and logs an ERROR naming the file — every tick, until a human fixes or
   removes it. Nothing is swallowed and nothing is retried silently.

Two ticks are kept apart by an advisory whole-file lock (`fcntl.flock` on POSIX,
`msvcrt.locking` on Windows) held for the run. The kernel releases it when the process
dies, so a killed tick leaves no stale lock — which a pid file could not promise. A busy
lock is an INFO and exit 0, not an error: the previous tick is still working.

## 9. The tick

```text
seed the storage layout      # so a fresh home has the config the user is told to edit
load config, check enabled
take the tick lock           # busy -> exit 0
check min_interval           # too soon -> exit 0
for each project in scope:
    preflight + scan         # section 7
    for each agent:
        resolve Context      # agent + model, no graph, no checkpointer, no MCP
        load config, rules, skills, prompts; compute the fingerprint
        drop the candidates the ledger says are not due
        build one RunContext (command="skill-daemon")
        for each candidate:
            gate -> record_gate_refusal + ledger row, or
            reserve the worst-case LLM cost, run the pipeline, read the report, record
    stamp the project inbox
stamp the tick
```

Notes on the decisions that are easy to get wrong:

- **The budget is spent against the worst case.** `llm_call_bound(1, max_plans, ...)` is
  subtracted *before* a thread runs, so a tick stops before it can exceed
  `max_llm_calls_per_tick` rather than after. The number actually spent is read back from
  the decision report and stored in the ledger for observability only.
- **`Context.create` is reused, not reimplemented.** It resolves agent and model exactly as
  the CLI does and touches neither the graph, the checkpointer nor MCP. The pipeline reads
  three fields from it.
- **A missing agent costs its own threads only.** Trajectories outlive the agents that
  recorded them; a renamed or removed agent produces a warning and its threads stay unmined,
  with **no** ledger row — the trajectory is fine, the configuration moved, and restoring
  the agent should make the threads mineable again rather than leaving them skipped forever.
- **`command="skill-daemon"`** goes into every decision report this component produces.
  Nothing in the pipeline branches on `RunContext.command`; it is a report label, and it
  makes a background verdict distinguishable from one a user asked for.
- **The results come from the decision report.** `_mine_thread` returns a `PlanTally`; the
  proposal paths and the real LLM cost come from the report the pipeline already writes
  (REPORT_VERSION 2). Reading an existing contract beat changing the pipeline to return more.

## 10. Notification

`notify.inbox` writes `<project state>/skill-evolver/daemon-inbox.json` (atomic: temp +
fsync + `os.replace`, 0600) holding the last tick timestamp, the last time the user was
shown the banner, and the recent proposals.

`Session.start()` calls `print_daemon_notice`, which counts the proposals actually on disk,
compares the inbox against `last_seen` and prints one line:

```text
Skill daemon: 3 proposals waiting (1 new) - /skill-review
```

Reading is **fail-open** — a missing, truncated or hand-edited inbox must never stop a
session from starting; the failure is logged, not raised. This is the one place in the
component where a broad `except` is correct, and it logs with a traceback rather than
passing silently. Writing is not fail-open: a tick that cannot record its own result says so.

## 11. Deployment

`resources/systemd/msagent-skill-daemon.service` is a `Type=oneshot` user unit, `.timer` is
`OnCalendar=hourly` with `Persistent=true` (catch up after a suspend or reboot) and
`RandomizedDelaySec=600` (do not have every machine call the model at :00). The unit runs `Nice=10`,
`IOSchedulingClass=idle` so a tick stays out of the way of interactive work, under
`NoNewPrivileges`, `PrivateTmp` and `ProtectSystem=full` (`/usr`, `/boot`, `/efi`, `/etc`
read-only). Not `strict`: that makes everything outside `/home` read-only as well, and the
daemon writes proposals into each project's working directory, which can live anywhere —
`/mnt/d` under WSL, `/srv`, `/data`. Both choices were checked against systemd 252 in WSL:
`systemd-analyze --user verify` is clean, and a transient `systemd-run --user` of the real
daemon under the same properties mined a project outside `/home`.

A unit reads neither the shell profile nor, reliably, a `.env`: `core/settings.py` loads
`.env` relative to the working directory. The API key that `config.llms.yml` names in
`api_key_env` would be missing under the timer, so the unit loads
`EnvironmentFile=-%h/.msagent/skill-daemon.env` (the leading `-` makes a missing file not
an error). Install the units with `install -m 644`, not `cp`: files on a Windows mount look
executable, and systemd warns about executable unit files.

`min_interval_seconds` must stay below the timer period minus `RandomizedDelaySec`. The
random delay is drawn anew for every run, so two consecutive hourly runs can start 49
minutes apart (10 minutes of delay plus 1 minute of `AccuracySec` on the first, none on the
second), and a minimum of 3600 would refuse about half of them as "too soon". The default
1800 leaves room for that and still caps a misconfigured every-minute timer at two ticks an
hour. `--watch` defaults to 3600 seconds for the same reason.

A timer that fires too often is harmless: `min_interval_seconds` makes the extra ticks exit
immediately as "too soon".

For a host without systemd (WSL without systemd enabled), `--watch` runs the same tick in a
sleep loop; it re-reads the config each time, so edits apply without a restart. This is the
documented fallback, not the recommended deployment.

## 12. Limitations and future work

- **The pool is re-parsed per (project, agent).** Each pair reads up to
  `cross_session_limit` (20) JSONL files per tick. That is stdlib JSON parsing and, at
  `max_threads_per_tick: 5`, cheap. A parse cache keyed by `(path, mtime_ns, size)` in the
  ledger is the obvious next step if it ever stops being cheap.
- **Quiescence is a heuristic, not a fact.** A user who walks away for 20 minutes and comes
  back to the same thread will have it mined mid-session, and the continued thread then
  re-mined under `content_change`. The alternative — a real session-end event — is a write
  path change the project rules forbid.
- **mtime does not survive copying files between machines**, inherited from the same choice
  `/skill-mine --since` makes.
- **`--watch` has no backoff.** It sleeps a fixed interval; a model endpoint that is down
  produces one failed thread per tick until `max_attempts` poisons the row.
- **No per-tick cost cap in currency**, only in LLM calls. The calls differ wildly in prompt
  size, so `max_llm_calls_per_tick` bounds the count, not the spend.
- **The banner counts proposals, not their quality.** It says how many are waiting; whether
  any of them is worth promoting is what `/skill-review` is for.
- **One ledger for all projects.** Simple and correct for one user on one machine; a shared
  `MSAGENT_HOME` across users would need the lock and the ledger to move per user.
