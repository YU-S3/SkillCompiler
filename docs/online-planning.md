# Online analysis and executable Skeletons

The formal runner enables `runtime.planning_enabled: true`. Each task follows the same path in Train, Val and Test. All model prompts are generic role instructions; domain predicates and tools are supplied by the environment, not by benchmark dispatch in the core.

## Construction and reuse

1. `Planner.prepare()` sends the public task, observation, tool schemas, optional `predicate_contracts()` descriptions and available Skill interfaces to the Analyzer. Private scores are unavailable.
2. The Analyzer returns either `direct` with a reason or an explicit goal contract, at most eight ordered operations, and final output bindings. Every operation states `execution_mode` (`programmatic` or `agent`); programmatic operations list their `allowed_tools`. Schemas live in `schemas.py` and `planning.py`.
3. The Host validates input availability, exact data-edge types, acyclic dependencies, prerequisite support and coverage of every declared goal effect. Preconditions must follow from task prerequisites or ancestor effects. Later or unrelated operations cannot establish an earlier prerequisite.
4. The Host re-checks executability (Sec. 3.2.2). A programmatic operation must meet all of these conditions:
   - typed input and output roles, with at least one output;
   - predicates listed in `predicate_contracts()`;
   - adapter verifiers for its preconditions and effects (and validator, if any);
   - a non-empty set of existing `allowed_tools`.

   Any failed condition downgrades the operation to `agent`; an agent operation is never upgraded. The `task_analysis` event records requested mode, final mode and reasons.
5. Each operation is retrieved through `Bank.retrieve(..., requirement=contract)`. Bounded role matching returns proofs; compatible Skills are ranked by lexical similarity plus measured utility (Eq. 3). Failure or exhausted proof search cannot authorize reuse.
6. The constructor chooses from compatible operation IDs or complete compatible Skeletons. Programmatic operations become `execution_mode: "programmatic"` nodes; agent operations become `dynamic` nodes whose chosen Skill is guidance only. Rebinding updates input roles, output consumers and node contracts together.
7. The resulting Skeleton is checked again. At runtime a programmatic node is handled in this order:
   1. It uses a unique eligible implementation (Bind, Pre, Env and Cert of Eq. 4).
   2. If none exists, it asks the task-local Programmer for one.
   3. If that also fails, it records `implementation_fallback` and continues under agent guidance.

   An eligible implementation that runs and fails is a localized node failure, not a silent retry. Runtime verifies actual preconditions, each executed node's effects and the complete goal's effects through the adapter. Only verified node/output events are marked successful. Native evaluation still determines task correctness.

A task-local Programmer request is validated before use when the pipeline supplies an isolated instance of the same task (`Runtime.run(..., sandbox=...)`):

1. The Host opens a fresh environment and replays the live native-tool prefix.
2. It runs the candidate there, including output-schema and effect checks.
3. Discrepancies go back to the Programmer for up to `runtime.validation_repair_limit` repairs.
4. Only a passing candidate receives a `task_local` certificate and runs against the live environment.

Each attempt is recorded as a `runtime_validation` event. Without an isolated instance only structural admission applies.

Planning has at most one structural repair. Invalid construction or unverifiable prerequisites leave an explicit trace and fall back to the ordinary bounded agent. Provider failures, changed request identity and exhausted model budgets remain explicit stopping conditions, not unlimited planning retries. All planning requests consume the ordinary run ledger.

This proves composition relative to an explicit symbolic contract. It does not prove that a model perfectly translated a natural-language goal, that arbitrary Python is equivalent, or that undeclared world effects cannot occur. Adapters must check observable state faithfully. The native evaluator remains independent.

## Contracts and persistent learning

A complete Skeleton has `contract`, ordered `nodes` and `outputs`. Each node declares its operation `contract`, `args`, and `after`. Skill/Program references must satisfy that operation. `task` bindings read public inputs; `from` bindings read declared ancestor outputs; `literal` carries a constant. A final `answer` role returns the task answer.

The extractor sees real Train analysis, construction, execution and feedback. It may propose persistent operations, Programs and complete Skeletons; it does not automatically publish every task-local plan. New Programs require source-backed replay. A failed replay goes back to the Programmer with the expected observation, the observed result and the previous source, for up to `learning.program_repair_limit` repairs. A passing Program receives a persistent certificate. Refinement uses later Train evidence, strict identity proofs, replacement regression and transactional consumer rewiring. Full Skeleton contracts are checked again when the candidate Bank is built. Frozen and Test-time overlays remain separate.

Learning material includes `failure_localization`: for each failed episode, the first failing operation in the trace. The levels are:

- `skeleton_node` or `skeleton_goal`, with the node binding and contract;
- `program` or `skill`;
- `task_local_program`, for failed isolated validation;
- `operation`;
- `task_outcome`, when no local operation failed but the evaluator rejected the answer.

After each training batch is learned, failed tasks of that batch are re-executed with the updated Bank (`training.replan_limit` rounds, Sec. 3.4.2). A re-executed task replaces its earlier record and keeps a `previous_attempt` summary (Bank hash, feedback and localization). Learning then runs again on the corrected evidence. Re-execution stops early when nothing failed or the Bank did not change.

Legacy deployment Skeletons lacking a complete goal contract remain loadable but are excluded from verified complete-Skeleton retrieval. Their Skills and Programs remain accessible through their normal interfaces. The framework does not fabricate missing semantic contracts from old experimental records. `planning_enabled: false` retains the legacy selection path for explicit compatibility checks; it is not the formal default.

## Adapter surface

In addition to ordinary public tools, an adapter may expose:

```python
predicate_contracts()  # public names, argument meanings, domains and supported checks
verify_preconditions(conditions, inputs)
verify_effects(effects, inputs, variables, result)
verify_contract(validator, inputs, result)
```

These inspect public execution evidence, never gold. Unsupported conditions return false. Semantic validators are adapter specifications, not arbitrary JSON Schemas. The synthetic `examples/key_value.py` demonstrates verification against actual tool observations.

## Paper-to-code mapping

| Mechanism | Main implementation |
|---|---|
| Operation decomposition and analysis | `planning.Planner`, `ANALYZER` |
| Executability judgement (Sec. 3.2.2) | `planning.executability`, node `execution_mode` |
| Operation construction and Skeleton organization | `CONSTRUCTOR`, `validate_analysis`, `skeleton_analysis` |
| Program/Skill retrieval and utility (Eq. 3) | `Bank.retrieve`, `semantic.match_requirement`, `learning.with_usage` |
| Reuse eligibility (Eq. 4) | `Runtime.skill/program/_bound`, `qualification` |
| Programming, validation and repair (Sec. 3.2.4) | `Programmer`, `Runtime._compile/_sandbox_validate`, `Learner._build/_trial`, `ProgramWorker` |
| Target agent and environment execution | `Runtime`, `ToolContext`, environment adapter |
| Failure localization and proposals (Sec. 3.4.1) | `learning.localize_failure`, `Learner.update`, `EXTRACTOR`, `REFINER` |
| Re-execution after an update (Sec. 3.4.2) | `Pipeline.run` (`replan_limit`) |
| Semantic alignment and merging | `semantic.align/verify`, `alignment.integrate` |
| Bank updating and validation selection | `Pipeline._checkpoint`, `Pipeline.run` |

The table describes implemented mechanisms. Historical accuracy still requires the original dataset/environment configuration and a new evaluation; portable Banks alone do not establish a reproduction.

## Checks

`tests/test_planning.py` exercises the default planning path, including multi-operation dependencies, directional retrieval, role rebinding, missing-effect rejection, bounded repair, actual Docker execution, task-local programming, and freeze/resume. `tests/test_mechanisms.py` drives fabricated tasks through each mechanism above: utility ranking, certificate/environment rejection and fallback, executability downgrade, sandbox validation with repair, training replay repair and certification, failure localization and re-execution. Existing component tests explicitly select legacy planning where their scripted responses test earlier independent interfaces. CI runs Python 3.10 and 3.12 plus a real container job.
