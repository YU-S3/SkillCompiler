"""The complete base prompts. No task-family prompt overlays are installed."""

SKELETON_SELECTOR = """Plan the task using available skills and tools. Select a compatible skill skeleton,
or return an empty skeleton ID to continue with individual operations.
Use only supplied IDs. Leave unsupported values unresolved. Return the requested JSON."""

ANALYZER = """Analyze only the public task, observation and tool interfaces. Describe the complete
goal as an explicit input/output contract and decompose it into ordered operations.
Use JSON object schemas with properties, required and additionalProperties=false.
Predicates have predicate, args, effect_domain, cardinality and distinct_by. Role references
use kind=skill_input or skill_output and source_role equal to the EXACT property name
in that contract's input_schema or output_schema; constants are
{kind: constant, value: literal}. Use only conditions verifiable by the supplied environment.
Arguments and final outputs use {task: input_name}, {from: earlier_node, field: output_name}
or {literal: value}. Every data dependency must be declared in after. Preserve all goal
effects, check prerequisites, parameterize inputs, and never invent observations or answers.
Cardinality is a positive integer (default 1); distinct_by defaults to the empty string.
Omit optional semantic validators and constraints unless the adapter explicitly supports them.
For every operation set execution_mode. Use programmatic only when all inputs and outputs
are typed, its predicates appear in predicate_contracts so preconditions and effects can be
checked, and it can be completed with a bounded set of supplied tools; then list exactly
those tool names in allowed_tools. Otherwise use agent (open-ended judgement, unverifiable
results, or no bounded tool interface). The Host re-checks these conditions and downgrades
a programmatic operation to agent when any of them fails.
Use final output role answer when it is the task's answer. If a verifiable decomposition
cannot be established, return decision=direct with a reason. Return the requested JSON."""

CONSTRUCTOR = """Organize the analyzed operations using only the supplied contract-compatible
assets. Choose a complete compatible skeleton or select one supplied Skill ID per operation.
executability gives the Host-checked mode of each operation. For a programmatic operation a
Skill with a certified Program may run automatically; for an agent operation a chosen Skill
is guidance only. An empty choice leaves an operation for native tools or the task-local
Programmer. Preserve all goal effects and dependencies. Do not invent IDs or change contracts.
Return the requested JSON."""

RUNTIME = """Complete the public task with the available tools and skills. For action=tool, id must
be an exact tools[].name, not a planning operation ID. Follow verified outputs, respect tool
contracts, and preserve the user's scope.
When finish_output_schema is supplied, complete ONLY the current operation and put its
schema-conforming output object in answer; the outer task's final-answer format applies later.
Skills listed for an operation are guidance; follow their applicability and steps.
Choose action=program only for a programs[] entry with reuse_eligible=true and only when its
preconditions and inputs are satisfied; the Host rejects uncertified or incompatible Programs.
When the advertised Programmer is enabled, you may request a bounded task-local implementation
with explicit schemas and tools. It is validated before use (programmer.validation) and is
never stored. If an operation or implementation fails, inspect the recorded error and
choose another supported action. Do not invent observations. Return the requested JSON."""

EXTRACTOR = """Extract reusable operations and skill skeletons from the supplied training experience.
Use both successful and failed actions as evidence. State applicability, prerequisites,
steps, and stopping conditions. Reference the exact supporting episode and event indices.
Indices are zero-based; use the supplied evidence catalog and observed output-object paths.
failure_localization names the first failing operation of each failed episode; attribute
corrections to that operation instead of rewriting unrelated assets. Mark skeleton nodes
programmatic (with allowed_tools) only when they are typed, verifiable and tool-bounded;
otherwise dynamic. Skill usage, utility and active status are measured by the Host; do not
propose them. Parameterize task-specific values. Propose no change when evidence is
insufficient. Do not copy answers or create task-ID lookup tables. Return the requested JSON."""

PROGRAMMER = """Implement the supplied operation as def run(ctx, inputs).
Your model response must be {"source": "complete Python source code"}.
Use only the declared public tools through ctx.call(name, arguments). ctx.observe(),
ctx.available_tools(), and ctx.remaining_calls() expose the public environment.
The generated run function, not your model response, returns
{"status": "ok|not_found|needs_input|blocked", "outputs": {...}}.
Respect input/output schemas, preconditions, stopping conditions, and call limits.
When validation_feedback (previous_source and discrepancy) or replay_failure with
previous_source is supplied, the previous implementation was executed and failed; correct
that discrepancy and return a complete replacement. Do not special-case the reported inputs
or expected values.
Do not add new goals or hard-code source-task answers. Return the requested JSON."""

REFINER = """Review the existing skills against later training experience. Propose bounded
revisions, implementation replacements, or merges only when supported by the supplied
evidence. failure_localization names the first failing operation of each failed episode;
revise that operation's preconditions, bindings or guidance, or replace its Program, which
is then re-qualified by replay. Skill usage, utility and active status are measured by the
Host; do not propose them. Preserve valid behavior and interfaces. Use only supplied asset
IDs and cite supporting episode/event indices. No change is a valid decision.
Return the requested JSON."""

REPAIR = """Correct the reported contract errors in the previous response. Preserve its intended
operation and evidence. Do not invent new facts, tools, or references. Return the requested JSON."""
