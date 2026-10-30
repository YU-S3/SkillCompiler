# Online analysis and executable Skeletons

The formal runner enables `runtime.planning_enabled: true`. Each task follows the same path in Train, Val and Test. All model prompts are generic role instructions; domain predicates and tools are supplied by the environment, not by benchmark dispatch in the core.

## Construction and reuse

1. `Planner.prepare()` sends the public task, observation, tool schemas, optional `predicate_contracts()` descriptions and available Skill interfaces to the Analyzer. Private scores are unavailable.
2. The Analyzer returns either `direct` with a reason or an explicit goal contract, at most eight ordered operations, and final output bindings. Schemas live in `schemas.py` and `planning.py`.
3. The Host validates input availability, exact data-edge types, acyclic dependencies, prerequisite support and coverage of every declared goal effect. Preconditions must follow from task prerequisites or ancestor effects. Later or unrelated operations cannot establish an earlier prerequisite.
4. Each operation is retrieved through `Bank.retrieve(..., requirement=contract)`. Bounded role matching returns proofs; lexical similarity ranks compatible Skills. Failure or exhausted proof search cannot authorize reuse.
5. The constructor chooses from compatible operation IDs or complete compatible Skeletons. Missing operations remain dynamic and may use native tools or the bounded Runtime Programmer. Rebinding updates input roles, output consumers and node contracts together.
6. The resulting Skeleton is checked again. Runtime verifies actual preconditions, each executed node's effects and the complete goal's effects through the adapter. Only verified node/output events are marked successful. Native evaluation still determines task correctness.

Planning has at most one structural repair. Invalid construction or unverifiable prerequisites leave an explicit trace and fall back to the ordinary bounded agent. Provider failures, changed request identity and exhausted model budgets remain explicit stopping conditions, not unlimited planning retries. All planning requests consume the ordinary run ledger.

This proves composition relative to an explicit symbolic contract. It does not prove that a model perfectly translated a natural-language goal, that arbitrary Python is equivalent, or that undeclared world effects cannot occur. Adapters must check observable state faithfully. The native evaluator remains independent.

## Contracts and persistent learning

A complete Skeleton has `contract`, ordered `nodes` and `outputs`. Each node declares its operation `contract`, `args`, and `after`. Skill/Program references must satisfy that operation. `task` bindings read public inputs; `from` bindings read declared ancestor outputs; `literal` carries a constant. A final `answer` role returns the task answer.

The extractor sees real Train analysis, construction, execution and feedback. It may propose persistent operations, Programs and complete Skeletons; it does not automatically publish every task-local plan. New Programs require source-backed replay. Refinement uses later Train evidence, strict identity proofs, replacement regression and transactional consumer rewiring. Full Skeleton contracts are checked again when the candidate Bank is built. Frozen and Test-time overlays remain separate.

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
| Operation construction and Skeleton organization | `CONSTRUCTOR`, `validate_analysis`, `skeleton_analysis` |
| Program/Skill retrieval | `Bank.retrieve`, `semantic.match_requirement` |
| Programming and qualification | `Programmer`, `ProgramWorker`, `Learner._trial` |
| Target agent and environment execution | `Runtime`, `ToolContext`, environment adapter |
| Trace analysis and proposals | `Learner.update`, `EXTRACTOR`, `REFINER` |
| Semantic alignment and merging | `semantic.align/verify`, `alignment.integrate` |
| Bank updating and validation selection | `Pipeline._checkpoint`, `Pipeline.run` |

The table describes implemented mechanisms. Historical accuracy still requires the original dataset/environment configuration and a new evaluation; portable Banks alone do not establish a reproduction.

## Checks

`tests/test_planning.py` exercises the default planning path, including multi-operation dependencies, directional retrieval, role rebinding, missing-effect rejection, bounded repair, actual Docker execution, task-local programming, and freeze/resume. Existing component tests explicitly select legacy planning where their scripted responses test earlier independent interfaces. CI runs Python 3.10 and 3.12 plus a real container job.
