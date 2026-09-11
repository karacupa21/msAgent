## description: Generates one SKILL.md from accepted candidates under the effective selection policy (contract_version 2)
## contract_version: 2

# SKILL.md generator

You write exactly one `SKILL.md` from knowledge candidates that a classifier accepted from a completed agent session. Each candidate is one durable rule: a title, an imperative rule, its expected future applicability and its target (a new skill, or an existing skill to update). When the classifier could establish them, a candidate also carries `When` (the condition under which the rule applies), `Constraints` (limits that must hold) and `Expected outcome` (what following the rule produced), plus `Evidence` lines: short cuts of the recorded events the rule was distilled from. You do not see the session, and you do not judge the candidates again: every accepted rule must be represented in the skill you write.

How to read an `Evidence` line:

- `user: …` — the task the user asked for, in the user's words.
- `tool.start <tool>: {…}` — one action with its real arguments, exactly as recorded.
- `tool.result <tool> (ok): …` — the real output of that action; this is the observed result.
- `tool.error <tool> (error): …` — a failure with the head and tail of its error text.
- `ai: …` — the agent's own statement; it is not proof of anything.
- `(N context excerpts omitted for the prompt budget)` — more context existed; do not guess what it said.

# Selection policy

{generation_policy}

# Task

- If the "Existing skill" section below contains a skill, reply with the **full revised text** of that skill: keep its `name`, integrate every rule, keep the durable content that still holds, and migrate the text to the required structure below.
- Otherwise create a new skill. Choose a durable kebab-case name that describes the task class or decision domain: lowercase letters, digits and hyphens only; never a ticket, PR or issue number, and never a name such as `fix-...`, `debug-...` or `audit-...` that only makes sense for one task. The selection policy above may prescribe a name prefix.
- State the `## Inputs` with their format (a CSV file with a header row, a column name, a path, an expected value); list the prerequisites the procedure depends on.
- Write concrete actions with the exact command or call shape the evidence shows, the expected result of each action and how to verify it. Never claim that a test or verification ran unless the evidence shows it ran; describe the check as something the future agent does.
- Replace one-time values (paths, ids, hosts, timestamps, the outputs of this run) with `<parameter>` placeholders and say what goes into them, but keep the durable file names, column names and environment conventions the procedure depends on.
- Never add a command, flag, API, version, unit, dependency or guarantee of success that the candidates and their evidence do not show.
- A genuinely one-operation procedure gets exactly one `## Workflow` step; do not pad it with filler steps.
- Place every rule where a future agent will act on it: a `## Workflow` step, a `## Constraints` entry, or the `## Inputs` / `## Outputs` sections. Phrase rules positively (what to do). A prohibition or safety constraint must come from the evidence (the failure it prevents is shown), never from general opinion about a tool.
- Use `When` as the trigger of the step or constraint, `Constraints` as `## Constraints` entries or conditions inside the step, and `Expected outcome` as the completion criterion of the step or in `## Outputs`. A candidate without these fields gets no invented condition: state the rule as given.
- Use the `Evidence` lines only to make the rule precise — the exact argument, flag, output or error class involved.
- Do not narrate the session. Do not include seq numbers, evidence ids such as `ev3`, thread ids, tickets, file names of one session, or any other one-time detail.

# Accepted candidates

{candidates}

# Existing skill

{existing_skill}

# REQUIRED SKILL.md STRUCTURE

Every newly created SKILL.md and every substantially rewritten SKILL.md must follow the structure below.

Existing skills should be migrated toward this structure when they are already being modified for a qualifying learning. Do not rewrite an otherwise unrelated skill solely to normalize its formatting.

The canonical structure is:

```markdown
---
name: <skill-name>
description: Use when <clear proactive trigger describing when this skill should be invoked>
---

# <Skill Title>

## Inputs

<What information, artifacts, files, context, state, tools, or prerequisites are expected before executing this skill.>

## Workflow

1. <Step>
2. <Step>
3. <Step>
...
n. <Step>

(one step when the procedure is one operation)

## Outputs

<What the skill is expected to produce, change, validate, or report.>

## Constraints

<Optional. Durable constraints, invariants, safety boundaries, environment limitations, or conditions that affect execution.>

## Examples

<Optional. Small reusable examples that clarify subtle usage or decision points.>
```

The following rules are mandatory.

## Description must be proactive

The frontmatter `description` MUST describe the trigger for invoking the skill.

It MUST begin with:

`Use when ...`

Write it as a proactive trigger, not as a summary of the file.

GOOD:

`description: Use when diagnosing build failures involving generated source or missing generated symbols.`

GOOD:

`description: Use when reviewing pull requests that modify database migrations or migration verification logic.`

BAD:

`description: Instructions for debugging build failures.`

BAD:

`description: This skill explains database migration reviews.`

BAD:

`description: Build troubleshooting skill.`

The description should help a future agent decide **whether to load the skill before starting the task**.

Prefer recognizable task conditions, symptoms, artifacts, or decision contexts.

Do not turn the description into a long keyword list.

Do not include session-specific names, ticket IDs, exact incidents, or temporary implementation details.

## Inputs is mandatory

Every SKILL.md MUST contain:

`## Inputs`

Inputs should state what the workflow expects to have available before or during execution.

Possible inputs include:

- user request;
- repository or working tree;
- error output;
- logs;
- source files;
- configuration;
- API schemas;
- generated artifacts;
- test results;
- environment metadata;
- tool availability;
- relevant support files;
- prior diagnostics.

Describe inputs by role, not by one-session filenames unless the filename is itself a durable repository convention.

GOOD:

> Build error output and the affected package or target.

BAD:

> Error from today's `foo.ts` failure.

Do not invent required inputs that the skill does not actually need.

## Workflow is mandatory

Every SKILL.md MUST contain:

`## Workflow`

Workflow must be an ordered numbered procedure.

Use this form:

```markdown
## Workflow

1. First step.
2. Second step.
3. Third step.
```

The workflow should encode executable decision logic, not a narrative description.

Each step should answer what the future agent should **do**.

Prefer imperative instructions.

GOOD:

1. Reproduce the reported failure before modifying source.
2. Identify whether the missing symbol is generated or handwritten.
3. Verify the generation step and toolchain version before changing implementation code.
4. Re-run the narrowest relevant verification target.

BAD:

1. The previous agent reproduced the problem.
2. It then noticed generated symbols were missing.
3. Eventually generation fixed the problem.

Do not preserve trajectory chronology unless that order is itself the reusable procedure.

### Workflow granularity

Include enough detail that the skill changes future behavior.

Do not expand routine actions into unnecessary micro-steps.

For example, avoid:

1. Open the terminal.
2. Change directory.
3. Run the command.
4. Read the output.

unless one of those actions contains a requirement that is easy to miss.

### Conditional workflow

Decision points may appear inside numbered steps.

Example:

```markdown
1. Reproduce the failure with the narrowest relevant command.
2. Determine whether the failing artifact is generated.
   - If generated, verify its producer step before editing consumers.
   - If handwritten, inspect the source directly.
3. Apply the smallest change that addresses the verified cause.
4. Re-run the narrow verification, then the broader suite if needed.
```

Keep the primary execution sequence numbered even when individual steps contain branches.

## Outputs is mandatory

Every SKILL.md MUST contain:

`## Outputs`

Outputs should define the observable result of applying the skill.

Possible outputs include:

- code changes;
- diagnosis;
- validated build;
- test result;
- review findings;
- generated artifact;
- configuration change;
- decision;
- report;
- updated documentation;
- verification evidence.

Outputs should describe completion criteria when possible.

GOOD:

> A reproduced and diagnosed failure, the smallest justified fix, and verification that the relevant target now passes.

BAD:

> The issue is fixed.

For review or diagnostic skills where no mutation is guaranteed, state that clearly.

Example:

> A ranked set of findings with supporting evidence; no repository mutation unless the task explicitly requests changes.

## Constraints is optional

Use:

`## Constraints`

only when the skill has durable constraints that materially affect execution.

Examples:

- required ordering;
- invariants;
- prohibited mutation scope;
- environment-specific boundaries;
- compatibility requirements;
- validation requirements;
- task-class-specific user preferences;
- conditions under which the workflow must stop or branch.

Constraints should primarily express durable boundaries.

Prefer positive formulations where possible. Prohibitions are allowed when the evidence shows the failure they prevent.

GOOD:

> Preserve generated files unless the repository explicitly treats them as checked-in source.

GOOD:

> Run schema generation before type checking when generated API types are build inputs.

Avoid weak negative folklore:

BAD:

> The type checker is broken before generation.

Do not create an empty `Constraints` section.

## Examples is optional

Use:

`## Examples`

only when a concise example materially improves correct application of the skill.

Examples are most useful for:

- ambiguous trigger conditions;
- workflow branches that are easy to miss;
- correct vs incorrect application;
- expected command or file patterns;
- representative inputs and outputs.

Examples must be reusable and class-level.

Do not preserve:

- session transcripts;
- today's ticket;
- exact one-time values;
- long debugging histories;
- irrelevant demonstrations.

Do not create an empty `Examples` section.


# Output contract

Reply with the complete `SKILL.md` and nothing else. It starts with the `---` frontmatter line. Do not wrap it in markdown fences (the fenced blocks above illustrate the structure, not the reply format), and write no prose before or after it.
