#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#    http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""/direct-skill-generation: distill a recorded session into a SKILL.md proposal.

The command never touches the analysed thread and never writes into the
skill library. Evidence is extracted from the thread's trajectory file by
code and classified by the LLM; a SKILL.md is generated from it and reviewed
by the LLM, validated by code again, and written as a proposal under
``<skills root>/.proposals/`` for a human to review and move. The per-thread
stages live in :mod:`msagent.skill_evolver.pipeline`, shared with
/skill-mine; this module parses the command line, loads config and prompts
and gathers the evidence.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from rich.markup import escape

from msagent.cli.bootstrap.initializer import initializer
from msagent.cli.handlers.session_history import load_history, trim_history
from msagent.cli.theme import console
from msagent.core.constants import SKILL_EVOLVER_CONFIG_FOLDER_NAME
from msagent.core.logging import get_logger
from msagent.skill_evolver.classify import strip_code_fence, strip_think_blocks
from msagent.skill_evolver.cli_args import POLICY_USAGE, CliArgsError, SharedFlags, take_shared_flag
from msagent.skill_evolver.config import (
    LEGACY_PACKAGED_PROMPT_FILE,
    STAGES,
    DirectSkillGenerationConfig,
    EffectiveRules,
    SkillEvolverConfigError,
    apply_overrides,
    effective_rules,
    load_skill_evolver_config,
)
from msagent.skill_evolver.features import DetectorNote, Episode, gate_decision
from msagent.skill_evolver.pipeline import (
    REJECTED,
    LazyLlm,
    PlanContext,
    PlanOutcome,
    PlanTally,
    RunContext,
    ThreadInput,
    activation_hint,
    bundle_preview,
    cited_threads,
    collect_episodes,
    draft_lines,
    gate_lines,
    generate_for_plan,
    load_prompts,
    print_policy_block,
    report_bundle,
    report_config_error,
    report_plans,
    run_thread,
    skill_library_snapshot,
    status,
    supporting,
    target_record,
)
from msagent.skill_evolver.prompts import PromptContractError, packaged_prompts_root, resolve_stage_prompt
from msagent.skill_evolver.report import decisions_dir
from msagent.skill_evolver.retrieval import BM25Index, SkillDoc
from msagent.skills.factory import Skill
from msagent.trajectory_recorder.export import (
    find_trajectory_file,
    resolve_trajectories_dir,
    workspace_filter,
)
from msagent.trajectory_recorder.reader import (
    Trajectory,
    TrajectoryReadError,
    load_trajectories,
    load_trajectory,
)

logger = get_logger(__name__)

# Newest trajectories of the agent mined for procedures shared across
# sessions: the packaged default, mirrored by config.CROSS_SESSION_LIMIT
# (evidence.cross_session_limit of config.skill.evolver.yml).
CROSS_SESSION_LIMIT = 20
# STAGES and DirectSkillGenerationConfig are re-exported from config (tests import them here).
__all__ = [
    "CROSS_SESSION_LIMIT",
    "DIRECT_USAGE",
    "STAGES",
    "DirectOptions",
    "DirectSkillGenerationConfig",
    "DirectSkillGenerationHandler",
    "PlanContext",
    "PlanOutcome",
    "PlanTally",
    "parse_direct_args",
]

_REJECTED = REJECTED
_DEPRECATION_HINT = "Use /skill-mine for trajectory-based generation."
_NO_MESSAGES_HINT = " ".join(
    (
        "Current thread has no messages.",
        "Use `/direct-skill-generation last` for the previous session.",
    ),
)
_DRY_RUN_DONE = "Dry run: nothing was written, no LLM was created, no decision report saved."
DIRECT_USAGE = "/direct-skill-generation [last|<thread-id>] [--dry-run] " + POLICY_USAGE

# The pipeline helpers under their historical names (mining and the tests reach them here).
_collect_episodes = collect_episodes
_supporting = supporting

# ---------------------------------------------------------------------------
# Legacy replay path: ``_generate_skill_md`` replays the whole session to the
# LLM with the ``prompts/<active>/`` variant. ``handle()`` no longer uses it;
# the helpers stay until the evidence pipeline has proven itself.
_HISTORY_BUDGET_RATIO = 0.6  # part of context window for session reply
# Output-contract sentinel: the model found nothing worth saving. A stray pair
# of backticks is tolerated because models copy the phrase as inline code.
_NOTHING_TO_SAVE = re.compile(r"^\s*`?\s*Nothing to save\.?\s*`?\s*$", re.IGNORECASE)
_REPLAY_SYSTEM_PROMPT = (
    "You are reviewing a COMPLETED agent session. The conversation that follows is a "
    "replay of that session for analysis only. Do not continue it and do not execute "
    "its tasks. After the replay you will receive exactly one instruction message; "
    "follow only that instruction."
)
_OMITTED_NOTE = " ".join(
    (
        "[Note: {n} messages from the middle of the session were omitted",
        "due to context limits.]",
    ),
)


@dataclass(frozen=True, slots=True)
class DirectOptions:
    """Parsed /direct-skill-generation arguments."""

    target: str | None = None
    policy: str | None = None
    demo: bool | None = None
    dry_run: bool = False


def parse_direct_args(args: list[str]) -> DirectOptions:
    """Parse ``[last|<thread-id>]`` plus the shared flags; every violation raises CliArgsError."""
    shared = SharedFlags()
    target: str | None = None
    index = 0
    while index < len(args):
        consumed = take_shared_flag(args, index, shared)
        if consumed is not None:
            index = consumed
            continue
        token = args[index]
        if token.startswith("--"):
            raise CliArgsError(f"unknown argument '{token}'")
        if target is not None:
            raise CliArgsError(f"unexpected argument '{token}'")
        target = token.strip() or None
        index += 1
    return DirectOptions(target=target, policy=shared.policy, demo=shared.demo, dry_run=shared.dry_run)


class DirectSkillGenerationHandler:
    """Generate a SKILL.md proposal from a recorded thread without mutating it."""

    def __init__(self, session) -> None:
        self.session = session

    async def handle(self, args: list[str]) -> None:
        """Distill a thread into a SKILL.md proposal: [last|<thread-id>] [--dry-run] [--policy ..] [--demo|--no-demo]."""
        console.print_info(_DEPRECATION_HINT)
        try:
            options = parse_direct_args(args)
        except CliArgsError as exc:
            console.print_error(escape(str(exc)))
            console.print(f"[muted]{escape(DIRECT_USAGE)}[/muted]")
            console.print("")
            return
        try:
            cfg = apply_overrides(self._load_config(), policy=options.policy, demo=options.demo)
            rules = effective_rules(cfg, working_dir=Path(self.session.context.working_dir))
            thread_id, messages = await load_history(self.session, options.target)

            if not messages:
                if options.target is None:
                    console.print_warning(_NO_MESSAGES_HINT)
                else:
                    console.print_warning(f"No messages found for thread '{thread_id}'")
                console.print("")
                return

            await self._run_pipeline(thread_id, cfg, rules, options)
        except (SkillEvolverConfigError, PromptContractError) as exc:
            report_config_error(console, exc)
            console.print("")
        except TrajectoryReadError as exc:
            console.print_warning(escape(f"Trajectory unreadable ({exc}); the LLM was not called"))
            console.print("")
        except Exception as exc:
            console.print_error(escape(f"Error generating skill: {exc}"))
            console.print("")
            logger.exception("Skill generation failed")

    # ------------------------------------------------------- evidence pipeline

    async def _run_pipeline(
        self,
        thread_id: str,
        cfg: DirectSkillGenerationConfig,
        rules: EffectiveRules,
        options: DirectOptions,
    ) -> None:
        """Trajectory → evidence → (dry run | pipeline.run_thread)."""
        ctx = self.session.context
        root = initializer.app_paths.home / SKILL_EVOLVER_CONFIG_FOLDER_NAME / "prompts"
        prompts = await load_prompts(self._load_stage_prompt, root, cfg)

        work = Path(ctx.working_dir)
        state_dir = initializer.get_project_paths(work).root
        trajectories_dir = resolve_trajectories_dir(state_dir=state_dir)
        workspace = workspace_filter(work)
        trajectory_path = find_trajectory_file(
            trajectories_dir,
            thread_id,
            workspace=workspace,
        )
        if trajectory_path is None:
            where = f"thread {thread_id} in {trajectories_dir}"
            msg = f"No recorded trajectory for {where}; the LLM was not called"
            console.print_error(escape(msg))
            console.print("")
            return

        skills = await self._load_skills()
        with self._status(f"Extracting evidence from thread {thread_id}..."):
            current, others, episodes, notes = await asyncio.to_thread(
                self._gather_evidence,
                trajectory_path,
                trajectories_dir,
                ctx.agent,
                skills,
                cross_session_limit=rules.cross_session_limit,
                demo=rules.demo,
                workspace=workspace,
            )
        thread = ThreadInput(current, others, episodes, notes)
        print_policy_block(console, cfg, rules)
        if options.dry_run:
            self._report_dry_run(thread, rules)
            return

        run = RunContext(
            command="direct-skill-generation",
            working_dir=work,
            state_dir=state_dir,
            requested=cfg,
            rules=rules,
            prompts=prompts,
            skills=skills,
            llm_slot=LazyLlm(self.session),
            taken={s.name for s in skills} | {s.display_name for s in skills},
            sink=console,
            report_dir=decisions_dir(state_dir) if rules.save_decision_report else None,
        )
        result = await run_thread(run, thread)
        for line in draft_lines(result):
            console.print(f"[muted]{escape(line)}[/muted]")
        tally = result.tally
        # The plan summary belongs to a thread whose plan loop ran; an early stop already said why.
        # A refused plan is a warning: the thread ran to its end and its report says why nothing was written.
        if result.stop_message is None and (tally.flagged or tally.plans > 1):
            line = f"Plans: {tally.proposals} proposals, {tally.describe()}"
            report = console.print_warning if tally.generation_errors else console.print_info
            report(line)
        sources = ", ".join(f"{stage}={source}" for stage, source in prompts.variants().items())
        console.print(f"[muted]Prompts: {escape(sources)}[/muted]")
        console.print("")

    @staticmethod
    def _report_dry_run(thread: ThreadInput, rules: EffectiveRules) -> None:
        """Gate decision and bundle preview; no LLM, no report, nothing written."""
        decision = gate_decision(thread.episodes, min_score=rules.min_evidence_score, demo=rules.demo)
        lines = [
            "Dry run: no LLM will be created.",
            *gate_lines(decision, min_score=rules.min_evidence_score, episodes=thread.episodes),
            *bundle_preview(thread, rules),
        ]
        for line in lines:
            console.print(f"[muted]{escape(line)}[/muted]")
        console.print_info(_DRY_RUN_DONE)
        console.print("")

    @staticmethod
    def _gather_evidence(
        trajectory_path: Path,
        trajectories_dir: Path,
        agent: str,
        skills: list[Skill],
        *,
        cross_session_limit: int = CROSS_SESSION_LIMIT,
        demo: bool = False,
        workspace: Path | None = None,
    ) -> tuple[Trajectory, list[Trajectory], list[Episode], list[DetectorNote]]:
        """Load the thread's trajectory plus the agent's newest; detect episodes.

        Returns the trajectory, the other trajectories its episodes cite (the
        evidence bundle must index them), the episodes and the detector notes
        (why observed_procedure candidates were dropped, demo only). In the
        shared store ``workspace`` keeps the newest to this workspace's threads.
        """
        current = load_trajectory(trajectory_path)
        newest = load_trajectories(
            trajectories_dir,
            agent=agent,
            workspace=workspace,
            limit=cross_session_limit,
        )
        others = [traj for traj in newest if traj.thread_id != current.thread_id]
        docs = [SkillDoc(skill.display_name, skill.description) for skill in skills]
        notes: list[DetectorNote] = []
        episodes = collect_episodes(current, others, skill_index=BM25Index(docs), demo=demo, notes=notes)
        return current, supporting(others, episodes), episodes, notes

    # The pipeline's helpers under their historical names (one implementation).
    _report_bundle = staticmethod(report_bundle)
    _cited_threads = staticmethod(cited_threads)
    _report_plans = staticmethod(report_plans)
    _generate_for_plan = staticmethod(generate_for_plan)
    _target_record = staticmethod(target_record)
    _activation_hint = staticmethod(activation_hint)
    _skill_library_snapshot = staticmethod(skill_library_snapshot)

    @staticmethod
    def _status(text: str):
        """Spinner shown while a pipeline stage runs."""
        return status(console, text)

    async def _load_skills(self) -> list[Skill]:
        """The skill catalogue of the current agent (rescanned, project dir first)."""
        ctx = self.session.context
        return await initializer.refresh_cached_skills(
            agent=ctx.agent,
            working_dir=ctx.working_dir,
        )

    async def _build_skill_library_snapshot(self) -> str:
        """Legacy replay path: the snapshot of the freshly loaded catalogue."""
        return skill_library_snapshot(await self._load_skills())

    @classmethod
    def _sanitize_model_output(cls, text: str) -> str:
        """Remove reasoning blocks the model may emit around the payload."""
        return strip_think_blocks(text).strip()

    # ------------------------------------------------------------------ config

    @staticmethod
    def _load_config() -> DirectSkillGenerationConfig:
        """Read config.skill.evolver.yml via config.load_skill_evolver_config; raises SkillEvolverConfigError."""
        return load_skill_evolver_config(initializer.app_paths.config_dir)

    # ----------------------------------------------------------------- prompts

    async def _load_stage_prompt(
        self,
        root: Path,
        cfg: DirectSkillGenerationConfig,
        stage: str,
    ) -> tuple[str, str]:
        """Prompt of one evidence-pipeline stage: prompts/<stage>/ (user root, then packaged; no glob)."""
        notes: list[str] = []
        prompt = await asyncio.to_thread(resolve_stage_prompt, root, cfg, stage, notes=notes)
        for note in notes:
            console.print_warning(escape(note))
        return prompt.text, prompt.source

    async def _load_prompt_template(
        self,
        root: Path,
        cfg: DirectSkillGenerationConfig,
    ) -> tuple[str, str]:
        """Legacy replay prompt: the explicit file of the ``active`` variant folder (no glob, no contract check)."""
        name = cfg.prompt_file or LEGACY_PACKAGED_PROMPT_FILE
        user = root / cfg.active / name
        packaged = packaged_prompts_root() / cfg.active / name
        path = user if user.is_file() else packaged
        if not path.is_file():
            raise ValueError(f"prompt file not found: {user} (nor packaged {packaged})")
        text = await asyncio.to_thread(path.read_text, encoding="utf-8")
        return text, str(path)

    # ------------------------------------------------- legacy replay (unused)

    @classmethod
    def _prepare_replay_messages(
        cls,
        messages: list[AnyMessage],
        llm,
        context_window: int | None,
    ) -> tuple[list[AnyMessage], int]:
        """Normalize and trim session messages so they replay as a valid chat."""
        normalized = [cls._normalize_replay_message(m) for m in messages]
        normalized = cls._drop_trailing_orphan_tool_calls(normalized)
        # head_tail: keep the task statement, cut the middle, keep the newest tail.
        return trim_history(
            normalized,
            llm,
            context_window,
            budget_ratio=_HISTORY_BUDGET_RATIO,
        )

    @staticmethod
    def _normalize_replay_message(message: AnyMessage) -> AnyMessage:
        """Fold private reasoning into visible text; strip fields APIs reject."""
        if not isinstance(message, AIMessage):
            return message
        kwargs = dict(message.additional_kwargs or {})
        reasoning = kwargs.pop("reasoning_content", None)
        if not reasoning:
            return message
        text = getattr(message, "text", None)
        if not isinstance(text, str):
            text = str(message.content)
        merged = f"<past_reasoning>\n{reasoning}\n</past_reasoning>\n\n{text}"
        return message.model_copy(
            update={"content": merged, "additional_kwargs": kwargs},
        )

    @staticmethod
    def _drop_trailing_orphan_tool_calls(
        messages: list[AnyMessage],
    ) -> list[AnyMessage]:
        """Drop a trailing tool call whose result never arrived (interrupted run)."""
        result = list(messages)
        while result:
            last = result[-1]
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                result.pop()
                continue
            break
        return result

    async def _generate_skill_md(
        self,
        messages: list[AnyMessage],
        template: str,
        thread_id: str,
    ) -> str:
        """Replay the real session messages to the LLM, then send the instruction."""
        ctx = self.session.context
        llm_config = await initializer.load_llm_config(ctx.model, ctx.working_dir)
        llm = initializer.llm_factory.create(llm_config)

        replay, omitted = self._prepare_replay_messages(
            messages,
            llm,
            llm_config.context_window,
        )

        instruction = template
        for placeholder, value in (
            ("{skill_library}", await self._build_skill_library_snapshot()),
            ("{agent}", ctx.agent),
            ("{thread_id}", thread_id),
            ("{working_dir}", str(ctx.working_dir)),
            # Older templates carried a history placeholder; the replay replaces it.
            ("{history}", "(the full session is replayed above)"),
        ):
            instruction = instruction.replace(placeholder, value)
        if omitted:
            instruction = f"{_OMITTED_NOTE.format(n=omitted)}\n\n{instruction}"

        payload: list[AnyMessage] = [
            SystemMessage(content=_REPLAY_SYSTEM_PROMPT),
            *replay,
            HumanMessage(content=instruction),
        ]

        response = await llm.ainvoke(payload)
        text = getattr(response, "text", None)
        if not isinstance(text, str):
            text = str(response.content)
        return self._strip_code_fence(self._sanitize_model_output(text))

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """Unwrap the whole answer if the model fenced it despite instructions."""
        return strip_code_fence(text)
