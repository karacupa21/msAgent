# msagent-skill-daemon

Background SKILL.md mining for msAgent.

msAgent records every session as a trajectory, and `/skill-mine` turns trajectories into
`SKILL.md` proposals — but only when someone remembers to type it. `msagent-skill-daemon`
runs the same pipeline on a schedule, with nobody at the keyboard. Each run (a *tick*)
finds the finished trajectories that have not been mined yet, mines them and records the
outcome, so no thread is ever paid for twice.

It never activates a skill. Everything it writes is an inactive proposal under
`.proposals/`, which you review and promote with `/skill-review` exactly as if `/skill-mine`
had produced it. It is **off by default**: an unattended run spends LLM calls.

Design and rationale: [ARCHITECTURE_skill_daemon.md](../../../../ARCHITECTURE_skill_daemon.md).

## Quick start

Run these from the repository root, in the environment msagent is installed in.

```bash
# 1. Install. The console script is new, so reinstall once.
bash build_install.sh                  # rebuild and reinstall the wheel
# ...or, in a development checkout:    uv pip install -e .

# 2. The first run creates ~/.msagent/config/config.skill.daemon.yml; then switch it on.
msagent-skill-daemon                   # -> skill-daemon is disabled; nothing was scanned
sed -i 's/^enabled: false/enabled: true/' ~/.msagent/config/config.skill.daemon.yml

# 3. See what it would mine. No LLM is created and nothing is written.
msagent-skill-daemon --dry-run -v

# 4. Mine for real.
msagent-skill-daemon -v
```

A dry run logs each thread it would mine, then prints a summary:

```text
... INFO msagent.skill_evolver.daemon.runner: skill-daemon: would mine thread-signals (score 3.00)
Tick over 1 projects: 1 threads, 0 proposals, 0 skipped by the gate, 0 nothing to save, 0 rejected at generation, 0 failed
```

A real tick ends with the same summary and the path of every proposal it wrote:

```text
Tick over 1 projects: 1 threads, 1 proposals, 0 skipped by the gate, 0 nothing to save, 0 rejected at generation, 0 failed
  proposal: /home/me/work/skills/.proposals/thread-signals/<skill-name>/SKILL.md
```

**Which threads get mined.** A trajectory is picked up once its file has been idle for
`quiet_period_seconds` (15 minutes by default — msAgent writes no "session ended" marker, so
silence is the signal) and its last turn finished instead of being interrupted. A thread is
mined again only if it grew (you continued the conversation) or the prompts or policy
changed. With `-v` the log says why each skipped file was skipped.

`python -m msagent.skill_evolver.daemon` works the same as the console script.

## Running on a schedule

### systemd user timer (Linux, and WSL with systemd enabled)

The units in [`resources/systemd/`](../../../../resources/systemd/) run one tick an hour;
`Persistent=true` catches up on runs missed while the machine was off.

```bash
# 1. Credentials. A systemd unit does not read your shell profile, so the API key named by
#    api_key_env in your config.llms.yml must come from a file the unit loads.
install -m 600 /dev/null ~/.msagent/skill-daemon.env
${EDITOR:-nano} ~/.msagent/skill-daemon.env     # e.g. OPENAI_API_KEY=...  HTTPS_PROXY=...

# 2. Install the units ("install -m 644", not "cp": files on a Windows mount look executable).
mkdir -p ~/.config/systemd/user
install -m 644 resources/systemd/msagent-skill-daemon.service \
               resources/systemd/msagent-skill-daemon.timer ~/.config/systemd/user/

# 3. Point ExecStart at your installation; it must be an absolute path.
sed -i "s|^ExecStart=.*|ExecStart=$(command -v msagent-skill-daemon) --once|" \
    ~/.config/systemd/user/msagent-skill-daemon.service

# 4. Start the timer.
systemctl --user daemon-reload
systemctl --user enable --now msagent-skill-daemon.timer
```

Day to day:

```bash
systemctl --user list-timers msagent-skill-daemon.timer   # when it ran last and runs next
systemctl --user start msagent-skill-daemon.service       # run a tick now
journalctl --user -u msagent-skill-daemon -e               # what the ticks printed
```

On a server, `loginctl enable-linger $USER` keeps the timer running while you are logged
out. In WSL the timer fires only while the distribution is running, and missed runs are
caught up when it starts; systemd itself is enabled with `systemd=true` under `[boot]` in
`/etc/wsl.conf`, followed by `wsl --shutdown`.

Installed from PyPI, without a checkout? The units ship inside the package:
`python -c "from importlib.resources import files; print(files('resources') / 'systemd')"`.

### Without systemd

```bash
# The same tick in a loop, once an hour (a container, a tmux pane, WSL without systemd):
nohup msagent-skill-daemon --watch >> ~/.msagent/logs/skill-daemon.log 2>&1 &

# ...or cron. It does not read your profile either, so export the env file first:
# 0 * * * *  set -a; . "$HOME/.msagent/skill-daemon.env"; set +a; /abs/path/to/msagent-skill-daemon >> "$HOME/.msagent/logs/skill-daemon.log" 2>&1
```

## Where the results are

| What | Where |
|---|---|
| **Proposals** — inactive `SKILL.md` plus `provenance.json` | `<project>/skills/.proposals/<thread-id>/<skill-name>/`, or under `generation.output_dir` when `config.skill.evolver.yml` sets one |
| **Why** a thread did or did not produce one | `~/.msagent/state/projects/<project-id>/skill-evolver/decisions/<thread-id>-<utc>.json`; reports from the daemon say `"command": "skill-daemon"` |
| Drafts refused by the validator or the reviewer | next to their report, `<report-name>.draft-<n>-<plan>.rejected.md` |
| What has been mined, thread by thread | `~/.msagent/state/skill-daemon/ledger.sqlite3` |
| What the startup banner reads | `~/.msagent/state/projects/<project-id>/skill-evolver/daemon-inbox.json` |
| The log | stdout and stderr; `journalctl --user -u msagent-skill-daemon` under systemd |

`<project>` is the working directory of the session that recorded the trajectory, and
`<project-id>` names its state directory. To print that directory for a project:

```bash
cd /path/to/project
python -c "from pathlib import Path; from msagent.core.paths import AppPaths; print(AppPaths.resolve().for_project(Path.cwd()).root)"
```

All `~/.msagent` paths move with `MSAGENT_HOME` if you set it.

With `output.scope: shared` in `config.trajectory.recorder.yml` every workspace records into one
`~/.msagent/state/trajectories/`. The daemon still mines each thread once, for the workspace it was
recorded in (the `working_dir` of the file's first event): the table above stays true project by
project, and the evidence pool is that workspace's own threads, as for `/skill-mine`. A file whose
workspace is out of scope or has no state directory any more is skipped with a log line.

### Reviewing proposals

Start `msagent` in the project. When the daemon has left something, the welcome screen
says so:

```text
Skill daemon: 3 proposals waiting (1 new) - /skill-review
```

Then, inside the session:

```text
/skill-review                  list the proposals (the DEMO column marks demo-mode ones)
/skill-review accept <name>    re-validate it and move it into the library: now active
/skill-review reject <name>    delete it, after a confirmation
```

The agent cannot see a proposal until you accept it.

### Inspecting the ledger

Without the `sqlite3` CLI:

```bash
python - <<'EOF'
import sqlite3
from pathlib import Path
db = Path.home() / ".msagent/state/skill-daemon/ledger.sqlite3"
for row in sqlite3.connect(db).execute(
        "SELECT thread_id, status, proposals, llm_calls, last_run FROM mined ORDER BY last_run DESC"):
    print(*row, sep="  ")
EOF
```

| `status` | Meaning |
|---|---|
| `mined` | at least one proposal was written |
| `gate_skip` | too little evidence to spend an LLM call on |
| `nothing` | the classifier found nothing worth a skill |
| `rejected` | every draft was refused (invalid twice, failed review, secrets, budget, transport error); see the `.rejected.md` drafts |
| `failed` | the thread raised an error; retried on the next tick |
| `poisoned` | failed `max_attempts` times in a row on the same file content; never retried |

Every status except `poisoned` is reconsidered when the thread grows or the methodology
changes. To force a thread to be mined again, delete its row from the `mined` table.

## Configuration

`~/.msagent/config/config.skill.daemon.yml`, created on the first run of `msagent` or
`msagent-skill-daemon` from the [packaged default](../../../../resources/configs/default/config.skill.daemon.yml).
A broken file stops the daemon with the offending key named; it never falls back to
defaults silently.

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Master switch |
| `schedule.quiet_period_seconds` | `900` | Idle time before a trajectory counts as finished |
| `schedule.min_interval_seconds` | `1800` | A tick sooner than this after the previous one exits as "too soon" |
| `schedule.max_threads_per_tick` | `5` | Threads mined per tick, across all projects |
| `schedule.max_llm_calls_per_tick` | `60` | The tick stops *before* it could exceed this |
| `scope.projects` / `project_dirs` | `all` / `[]` | `listed` restricts the daemon to the working directories in `project_dirs` |
| `scope.agents` | `[]` | Only these agents' trajectories; empty means all |
| `mining.model` | `null` | `null` uses the model msagent would; an alias from `config.llms.yml` puts background mining on a cheaper one |
| `mining.policy` / `mining.demo` | `strict_knowledge` / `false` | Same meaning as `/skill-mine --policy` and `--demo` |
| `mining.remine_on` | both | `content_change`, `methodology_change`: when a known thread is mined again |
| `mining.max_attempts` | `3` | Consecutive failures before a thread is poisoned |
| `notify.inbox` | `true` | Feed the startup banner |

Everything about the pipeline itself — the evidence gate, plans per thread, the per-thread
LLM limit, `output_dir`, the prompts — stays in `config.skill.evolver.yml`, shared with
`/skill-mine`.

Environment: `MSAGENT_SKILL_DAEMON_ENABLED=1` opts in without editing the file,
`MSAGENT_SKILL_DAEMON_DISABLED=1` is a kill switch that always wins, and
`MSAGENT_SKILL_DAEMON_CONFIG=<path>` reads another file. Both flags take effect on the next
tick, no restart needed.

## Command line

```text
msagent-skill-daemon [--once | --watch [SECONDS]] [--dry-run] [--force] [--project DIR_OR_ID] [-v]
```

| Option | Effect |
|---|---|
| `--once` | One tick, then exit (the default) |
| `--watch [SECONDS]` | Tick, sleep, repeat; 3600 seconds unless given |
| `--dry-run` | Select and gate threads only: no LLM, no ledger rows, no proposals |
| `--force` | Ignore `min_interval_seconds` for this run |
| `--project DIR_OR_ID` | Only this project, by working directory or project id |
| `-v` | Log at INFO to stderr |

Exit codes: `0` the tick ran, or was disabled, busy or too soon; `2` the configuration is
broken; `1` anything else. A thread that fails inside a healthy tick is a warning, not a
failed run — its ledger row and decision report hold the details.

## Troubleshooting

| You see | What it means |
|---|---|
| `skill-daemon is disabled; nothing was scanned` | `enabled` is `false`, or `MSAGENT_SKILL_DAEMON_DISABLED` is set |
| `too soon after the previous tick (…)` | Within `min_interval_seconds` of the last tick; `--force` overrides it once |
| `0 threads` although you have sessions | Rerun with `-v`: each file is logged as `still active`, `last turn has no turn.end` (interrupted with Ctrl+C), `out of scope`, or `not due (already mined)` |
| `skipping agent X ...; its threads stay unmined` | The trajectory's agent no longer exists in your agents config; restore it and its threads are picked up again |
| `unreadable trajectory in the pool of agent X` | A corrupt `.jsonl` in that project's `trajectories/` (with `output.scope: shared`, anywhere in `~/.msagent/state/trajectories/`); move it away. Logged on every tick until then |
| `deferring project ... holds no events yet` | A session had just created its file; it resolves itself on the next tick |
| `error: config.skill.daemon.yml: <key>: ...`, exit code 2 | Fix the key named |
| LLM authentication errors under systemd only | The unit cannot see the API key; see the credentials step above |
| `msagent-skill-daemon: command not found` | Reinstall (Quick start, step 1), or use `python -m msagent.skill_evolver.daemon` |
