# Bank format

`bank.json` has three arrays: `skills`, `programs`, and `skeletons`. IDs are unique across all three arrays and contain letters, digits, underscores, or hyphens. The loader rejects dangling references and Program source paths outside the Bank directory.

## Skills and Programs

A Skill has `id`, `name`, `guidance`, and optional applicability, input/output schemas, preconditions, effects, and `program_ids`. Guidance is operational content; provenance and execution receipts belong in the run output, outside a deployment Bank.

A Python Program has `id`, `name`, `format: "python"`, `source_file`, `input_schema`, `output_schema`, and an explicit `allowed_tools` list. Its source defines `run(ctx, inputs)`. Source is loaded into the canonical in-memory representation; the Bank digest covers executable source as well as guidance and structure.

## Skeletons

Nodes are in topological order. Each has an `id`, an `after` list, and an optional `skill_id` or `program_id`. A node without a Program executes through the agent using its goal and referenced skill guidance. `execution_mode: "dynamic"` explicitly selects this mode even when a Skill has an implementation. Intermediate outputs must satisfy the node's output schema.

Input bindings are explicit:

```json
{
  "id": "second",
  "after": ["first"],
  "program_id": "p",
  "args": {
    "original": {"task": "argument"},
    "computed": {"from": "first", "field": "result.value"},
    "limit": {"literal": 8}
  }
}
```

`task` references the public inputs plus `goal` (the task instruction), augmented by explicit planner bindings. `from` can reference only a previous node. Values are never evaluated as code. A skeleton may declare final `outputs`; an `answer` output completes the agent turn. Native scoring still determines success.

## Declarative Programs

Verified online reuse additionally requires a complete Skeleton `contract` and operation contracts on its nodes (or the referenced assets). Goal coverage and dependency checks run on Bank admission and again with public task inputs. Legacy incomplete Skeletons are not automatically promoted to this representation; see [online planning](online-planning.md).

`format: "ir"` stores `instructions` instead of Python source. Supported operations are `IF`, `FOR_EACH`, `ACTION`, and `RETURN`, with finite loop and action limits. References use declared inputs or local variables. Conditions support existence, equality, conjunction, disjunction, and negation.

Domain semantics belong to the adapter:

- `query(spec, inputs, variables)` resolves queries against public action catalogs or public evidence.
- `verify_effects(effects, inputs, variables, result)` confirms declared effects from observable state.
- `condition(spec, inputs, variables)` may handle additional explicit condition types.

The interpreter never guesses domain predicates. A missing verifier or unconfirmed effect rejects execution. Tool permissions and the shared native-call budget apply to every action. An `ACTION` may use `result_variable` to store its real tool response for later local references; `RETURN.output_sources` binds those observed values to declared outputs.

## Freeze and portability

`Bank.write()` writes only deployment content. It does not mutate the in-memory Bank. The run directory stores evidence, qualification receipts, parent checkpoints, and evaluation records separately. Keep that run directory when you need scientific provenance; a deployment Bank alone is not an experiment record.

Loading a Bank checks structural integrity. It is not a declaration that every Program is applicable or safe for every environment. Container isolation and adapter permissions remain required at execution time.
