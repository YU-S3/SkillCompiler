# Semantic alignment and merge transactions

The pipeline separates retrieval, contract identity, executable qualification and publication. Similar words can retrieve an asset; they cannot prove it interchangeable. These generic mechanisms adapt the original contract canonicalizer, bounded identity matcher and typed repair transaction to the portable Bank interface.

## Contracts and proofs

A semantic Skill declares `input_schema`, `output_schema`, `preconditions`, `effects` and optional `validator`/`entry_constraints`. JSON Schema property annotations can retain semantic types and resolution requirements. A predicate contains `predicate`, `args`, `effect_domain`, `cardinality` and `distinct_by`. Role references are explicit:

```json
{"predicate":"observed_value","effect_domain":"evidence","cardinality":1,
 "args":{"query":{"kind":"skill_input","source_role":"key"},
         "result":{"kind":"skill_output","source_role":"value"}}}
```

`semantic.align(source, target)` searches bounded input/output role bijections and returns `exact`, `different` or `unknown`. An exact result includes the role maps and contract hashes; `semantic.verify()` checks the proof independently. Predicate argument positions, domain, multiplicity, distinctness, schema constraints, validators and constants remain part of identity. Typed input/output references are renamed; literal strings and nested object property names are not.

The search cap is 4,096 states. Exhaustion, unknown role references and unsupported composed top-level interface schemas return `unknown`, never equivalence. Text-only assets without a semantic effect or validator can still provide guidance but cannot be automatically merged as contract-equivalent operations. This is an explicit contract proof, not a proof that arbitrary prose or Python programs mean the same thing.

Retrieval uses multiset Dice similarity after directional contract filtering when a requirement is supplied. Skills are then ranked by `Dice + 0.1·utility + 0.05·[active]` (Eq. 3), with deterministic ID tie-breaking. The Host measures utility from local operation outcomes; Skeletons are ranked lexically. The default online Planner supplies each analyzed operation's requirement. `match_requirement()` proves a role bijection, equal interface constraints, sufficient candidate effects and no additional unmet prerequisites. Extra effects or greater cardinality may satisfy a requirement; they do not establish merge identity. Validators and entry constraints must match exactly. Direct-agent guidance retrieval remains lexical.

Runtime implementation selection checks each binding against Eq. 4:

- the actual inputs bind and satisfy the schema;
- preconditions hold;
- every permitted tool exists in the current environment;
- the Program certificate still matches its body and tool interfaces.

Rejected bindings are recorded with reasons. Automatic invocation needs exactly one eligible implementation, or the node's preferred one. Several eligible implementations require an explicit choice rather than silently choosing the first.

## Aligning an implementation

New equivalent operations reuse the existing Skill interface. Program code keeps its original ABI; the binding carries the role conversion:

```json
{"program_id":"lookup_program",
 "input_map":{"key":"query"},
 "output_map":{"result":"value"}}
```

`input_map` maps Program inputs to Skill inputs. `output_map` maps Skill outputs to Program outputs. Legacy `program_ids` use identity mappings. Skeleton argument keys, downstream output references, node contracts and dynamic-node output schemas are retargeted consistently. Literals and nested output suffixes retain their meaning.

## The merge path

Extractor/refiner proposals may contain a replacement asset and an explicit operation:

```json
{"kind":"skills","source_ids":["lookup_a","lookup_b"],"target_id":"lookup_merged"}
```

1. Resolve all sources against the parent Bank. Require refinement mode, at least two independent Train episodes and a new target ID. Unknown sources, duplicate sources, overlapping edits and unqualified deletion are rejected.
2. Prove each source Skill contract equivalent to the replacement. A Program merge requires exact executable content and interfaces; a Skeleton merge requires the same ordered structure, dependencies and data bindings after node-ID normalization.
3. Transfer validated implementation bindings through the proven role maps. A merge cannot introduce an unrelated mapping. Rewrite every affected Skeleton consumer and check all references.
4. Replay the existing executable source cases in reset environments, using outputs from actual observations. Program replacement must pass its newly proposed cases and all available successful historical source cases, preserving the public ABI, permissions and semantic contract. Missing exact replay boundaries are rejected. Text-only merges require evidence that their guidance was used; Skeleton merges require executed source outputs.
5. Construct and validate the entire candidate Bank before committing its checkpoint. Any failed check leaves the original Bank unchanged. Receipts include source/target IDs, parent content hashes, alignment proofs, replay outcomes and the new content hash. Historical Bank checkpoints remain intact.

The deployment Bank contains guidance, contracts, bindings, executable sources and compact Program certificates. Proofs, lineage and full replay evidence stay in the run's learning receipts. A proposal that resubmits a Program with an unchanged executable body is a certification, not a revision: only the certificate is updated. Validation selects among resulting checkpoints; neither Val nor Test feedback is supplied to the learner. In-place Skill revisions preserve role names and implementation bindings. Executable replacement uses the Program revision path and its mandatory replay gate. A structural or contract-changing edit outside the supported replacement rules must be submitted as a distinct candidate, rather than overwriting existing consumers silently.

## Runtime Programmer during Test

`runtime.allow_programmer` defaults to `true`; a pinned isolated worker is required. The default allows two compilation requests per task, one structural repair per compilation and one repair after isolated validation (`validation_repair_limit`, at most three). These limits apply identically to Train, Val and Test and are included in run identity.

The Programmer is reached in two ways: through the agent's `programmer` action, or automatically for a `programmatic` Skeleton node without an eligible implementation. Either way it receives a public operation description, input/output schemas, a tool allowlist and the actual inputs. The Programmer sees the public task, observation, available permitted tools and observable execution history. It receives no private labels or evaluator feedback. Unknown tools and unresolved preconditions fail before compilation.

The Host validates source structure and interface. When the pipeline provides an isolated instance of the same task, the following steps run before any live use:

1. Replay the live native-tool prefix in that instance.
2. Execute the candidate there, with schema and effect checks.
3. On failure, return the discrepancy and previous source to the Programmer (`validation_feedback`) for bounded repair.

A prefix that cannot be replayed, or exhausted repairs, rejects the candidate. A programmatic node then falls back to agent guidance. A passing candidate receives a `task_local` certificate and runs once against the live environment. Structural admission is not empirical promotion. Live schema/effect verification is recorded separately, and a failed live execution is not silently retried. The agent can inspect its observed result and choose a subsequent supported action within the same task budget.

Generated source, source hash, request key, repair counts, `runtime_validation` attempts and the live qualification outcome are saved in the task trace. The Program belongs to that task's overlay; it is not published to the persistent Bank, transferred to another Test task or fed into training. The Frozen hash remains unchanged. All model calls use the same durable request ledger and finite transport retry policy. Provider failures and exhausted model budgets propagate rather than becoming additional repairs or agent fallbacks.
