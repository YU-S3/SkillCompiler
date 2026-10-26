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

Retrieval uses multiset Dice similarity with deterministic ID tie-breaking. An explicitly supplied requirement additionally filters Skills by exact semantic contract. Runtime implementation selection validates actual inputs and preconditions; multiple ready implementations require an explicit choice rather than silently choosing the first.

## Aligning an implementation

New equivalent operations reuse the existing Skill interface. Program code keeps its original ABI; the binding carries the role conversion:

```json
{"program_id":"lookup_program",
 "input_map":{"key":"query"},
 "output_map":{"result":"value"}}
```

`input_map` maps Program inputs to Skill inputs. `output_map` maps Skill outputs to Program outputs. Legacy `program_ids` use identity mappings. Skeleton argument keys, downstream output references and dynamic-node output schemas are retargeted consistently. Literals and nested output suffixes retain their meaning.

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

The deployment Bank contains guidance, contracts, bindings and executable sources. Proofs, lineage and replay evidence stay in the run's learning receipts. Validation selects among resulting checkpoints; neither Val nor Test feedback is supplied to the learner. In-place Skill revisions preserve role names and implementation bindings. Executable replacement uses the Program revision path and its mandatory replay gate. A structural or contract-changing edit outside the supported replacement rules must be submitted as a distinct candidate, rather than overwriting existing consumers silently.

## Runtime Programmer during Test

`runtime.allow_programmer` defaults to `true`; a pinned isolated worker is required. The default allows two compilation requests per task and one structural repair per compilation. These limits apply identically to Train, Val and Test and are included in run identity.

The agent's `programmer` action supplies a public operation description, input/output schemas, tool allowlist and actual inputs. The Programmer sees the public task, observation, available permitted tools and observable execution history. It receives no private labels or evaluator feedback. Unknown tools and unresolved preconditions fail before compilation.

The Host validates source structure and interface, then runs one actual bounded execution in Docker. Structural admission is not empirical promotion. Live schema/effect verification is recorded separately; a failed live execution is not silently retried by the structural repair mechanism. The agent can inspect its observed result and choose a subsequent supported action within the same task budget.

Generated source, source hash, request key, repair count and qualification outcome are saved in the task trace. The Program belongs to that task's overlay; it is not published to the persistent Bank, transferred to another Test task or fed into training. The Frozen hash remains unchanged. All model calls use the same durable request ledger and finite transport retry policy. Provider failures and exhausted model budgets propagate rather than becoming additional structural repairs.
