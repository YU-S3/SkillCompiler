# SkillCompiler

**Reusable skills, executable programs, and ordered skill skeletons for tool-using agents.**

SkillCompiler turns execution experience into a persistent Bank of reusable operations. An agent solves training tasks with public tools, an extractor identifies reusable behavior, a programmer implements bounded operations, and a refinement stage updates skills using subsequent experience. Held-out validation selects a Bank; testing uses an immutable snapshot.

The core is task-agnostic. Dataset loading, environment actions, and scoring live behind a small adapter interface. Prompts are the base role instructions in [`prompts.py`](src/skillcompiler/prompts.py), with no benchmark-specific prompt overlays, answer heuristics, or task-name dispatch in the learning and execution pipeline.

## Architecture

```mermaid
flowchart LR
    T[Training tasks] --> A[Task analyzer and agent]
    B[(Skill Bank)] --> A
    A --> E[Execution traces and feedback]
    E --> L[Operation and skeleton extraction]
    L --> P[Programmer]
    P --> Q[Isolated replay qualification]
    L --> R[Skill refinement]
    Q --> B
    R --> C[Contract alignment and transactional merge]
    C --> B
    B --> V[Validation selection]
    V --> F[Frozen Bank]
    F --> X[Read-only testing]
    X --> LP[Task-local Programmer]
```

- **Skills** describe applicability, prerequisites, procedure, and stopping conditions. The Host measures each Skill's local success rate (utility) and ranks contract-compatible Skills by similarity plus utility.
- **Programs** implement operations with explicit schemas and public-tool permissions. Python runs in a network-isolated container; declarative programs use bounded control flow. A Program is invoked automatically only when its inputs bind, its preconditions hold, its tools exist and its validation certificate still matches its body and tool interfaces.
- **Skill skeletons** connect operations through ordered dependencies and explicit input/output bindings. Each node is `programmatic` (run a certified implementation, else compile and validate one, else fall back to the agent) or `dynamic` (the agent carries out the operation, with a Skill as guidance).
- **Online task analysis** creates a public goal contract and an operation DAG, with an executability judgement per operation that the Host re-checks. The Host checks prerequisites, data types, dependency order and complete goal-effect coverage before constructing or reusing a Skeleton. Contract-compatible retrieval produces role-rebinding receipts; unresolved planning falls back to the bounded native agent.
- **Contract alignment** proves role equivalence from explicit schemas, predicates, effects, cardinality and validators. It returns input/output role maps; names or similar guidance cannot prove identity.
- **Refinement** works from localized failures. It stages evidence-backed revisions, implementation replacements and merges, replays affected executable cases with bounded Programmer repair, rewires consumers, and commits a new Bank checkpoint only after all checks pass. Failed training tasks are re-executed with the updated Bank. Both successful and failed training episodes remain available.
- **Runtime programming** is available during Train, Val and Test. A generated Program is first validated against a replay of the task's tool prefix in a separate environment instance, with bounded repair, and then executes in the same isolated worker under the task's native-call and model budgets. Test-time Programs stay in a task-local overlay and never modify the frozen persistent Bank.

## Installation

Python 3.10 or newer is required. Program execution additionally requires Linux or WSL2 with Docker.

```bash
git clone https://github.com/YU-S3/SkillCompiler.git
cd SkillCompiler
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Run the deterministic, zero-HTTP integration example:

```bash
python -m examples.offline_demo --output runs/offline-demo
```

This example exercises native tools, extraction, validation selection, freezing, testing, and checkpoint reuse with a scripted model. Its scores are fixture behavior, not model evaluation results.

For a live check against a real endpoint, [`examples/inventory_live.yaml`](examples/inventory_live.yaml) runs the full pipeline on fabricated two-tool tasks ([`examples/inventory.py`](examples/inventory.py)) that need a price lookup, a stock lookup and a computed product.

Online contract planning is enabled by default. See [planning and execution](docs/online-planning.md) for the full Task Analyzer → contract retrieval → Skeleton → Programmer/Executor path and its adapter requirements.

## Train, validate, and test

Copy [`configs/default.yaml`](configs/default.yaml) and set the adapter, model endpoint, model name, and budget. Provide the key through the named environment variable:

```bash
export MODEL_API_KEY='your-key'
skillcompiler run --config configs/default.yaml --output runs/experiment
```

The default endpoint is a placeholder. The model transport expects an OpenAI-compatible, non-streaming chat-completions response with JSON content. Explicit provider generation parameters belong in `model.parameters`; there are no model-name branches. Provider-specific thinking or output-limit behavior must be configured and checked for that provider before a paid run.

The pipeline:

1. Checks split disjointness and records the public input identities. Only the training order is shuffled.
2. Analyzes each public task, validates the proposed operation dependencies and goal coverage, retrieves compatible operations, and constructs or reuses a complete Skeleton. It solves each training batch with the current Bank and records actual tool execution and evaluator feedback.
3. Extracts proposed updates, with the first failing operation of each failed episode localized in the learning material. New persistent Programs must reproduce declared outputs from observed training events in an isolated replay environment. Replay discrepancies go back to the Programmer for bounded repair, and passing Programs receive a certificate. One structural repair of a proposal is allowed by default. Contract-equivalent operations align to existing roles; explicit merges require independent Train evidence, exact contract proof and source replay for executable implementations.
4. Re-executes the batch's failed training tasks with the updated Bank (`training.replan_limit`) and learns again from the new attempts.
5. Applies the configured number of Train-only refinement rounds.
6. Evaluates unique Bank checkpoints on validation tasks. A candidate must exceed the current score by more than `min_gain`; ties retain the earlier Bank. Validation feedback never enters extraction or refinement.
7. Freezes the selected Bank and evaluates the Test split. The Bank digest must remain unchanged. The agent, or a programmatic node without an eligible implementation, can call the Programmer. Its source, isolated validation, structural admission and live execution result are recorded in that task's trace, without publishing a new persistent asset.

To resume, use the same command, output directory, source, configuration, and inputs. Completed task and learning checkpoints are verified and reused. A changed run identity is rejected. One process may own an output directory at a time.

### Enable Programs

Choose a locally installed Python image and pin its immutable digest:

```bash
docker pull python:3.12-slim
docker image inspect python:3.12-slim --format '{{index .RepoDigests 0}}'
```

Put the returned digest in the configuration:

```yaml
worker:
  image: python@sha256:YOUR_LOCAL_IMAGE_DIGEST
  timeout: 60
  memory_mb: 1024
```

Programs cannot execute directly on the host. The worker disables networking, mounts its source read-only, drops capabilities, runs as a non-root user, and enforces time, memory, process, and RPC limits. Install any required libraries in your own image before pinning its digest. Native tool effects remain the adapter's responsibility.

The same worker is used for on-demand Runtime programming in all splits. Configure `runtime.allow_programmer`, `max_programs_per_task`, `programmer_repair_limit` and `validation_repair_limit`, and `learning.program_repair_limit`, explicitly when changing that policy. See [semantic alignment, merge transactions and task-local programming](docs/semantic-alignment.md) for proofs, role bindings, admission gates and trace records.

## Bring an environment

Implement a `TaskSuite` and point `adapter.factory` to `your_module:factory`, or `path/to/adapter.py:factory` relative to the configuration file. The factory returns an object with:

```python
tasks(split)                 # list[Task]; public inputs only
open(task)                   # fresh, isolated Environment
score(task, answer, env)     # Feedback; native evaluator
```

An Environment exposes `observe()`, `tools()`, `call(name, arguments)`, and `close()`. Each tool declares an input schema and may declare an output schema. The learner reopens environments to replay qualification cases, so reset behavior must be deterministic or explicitly controlled by the adapter.

Keep private answers and evaluator state out of `Task.inputs`, observations, and tool results. Use the native evaluator in `score()`. See the complete synthetic adapter in [`examples/key_value.py`](examples/key_value.py) and the [interface guide](docs/interfaces.md). Real dataset loaders, data downloads, and benchmark evaluators are supplied separately.

## Portable Banks

The [`benchmark/`](benchmark/) directory contains deployment content organized by benchmark:

| Directory | Skills | Programs | Skeletons |
|---|---:|---:|---:|
| `alfworld` | 9 | 9 | 6 |
| `searchqa` | 7 | 4 | 1 |
| `officeqa` | 37 | 5 | 0 |
| `docvqa` | 11 | 1 | 0 |
| `livemath` | 4 | 3 | 2 |
| `spreadsheetbench` | 7 | 0 | 0 |

Each directory contains `bank.json` and any required Program source files. It contains skill guidance, executable contracts, and references, without experiment logs, model identifiers, seeds, or training histories. Text-only Banks remain text-only; missing executable assets are not fabricated. Domain knowledge in these Banks is data, not an overlay in the core prompts.

```bash
skillcompiler check-bank benchmark/alfworld
skillcompiler evaluate --config configs/my-environment.yaml \
  --bank benchmark/alfworld --split test --output runs/evaluation
```

These are portable deployment assets, separate from the empty-Bank training example. Using them requires an adapter that implements their declared tools and inputs. The declarative Bank additionally requires the semantic-query and effect-verification interfaces described in the [Bank guide](docs/bank-format.md). These Banks carry no validation certificates, because certificates are bound to a concrete tool interface. Their Programs therefore act as reference implementations, with agent-guided execution, until training replay in your environment certifies them. This repository does not assert that the refactored runner reproduces historical experiment scores without a new evaluation.

## Records and resource accounting

Each run stores public task traces, training feedback, learning receipts, Bank checkpoints, selection scores, and final freeze hashes. Model requests preserve the actual payload, response, and raw usage; API keys are never written to receipts.

Every HTTP attempt requires budget admission. Exact returned usage is counted when available; interrupted or unmetered attempts retain their reservation. Retries are finite and consume the same run budget. Unknown usage is not silently treated as zero. The implementation is intentionally single-controller; use separate output directories and ledgers for independent runs.

## Verification

```bash
pytest -q
export SKILLCOMPILER_TEST_IMAGE='python@sha256:YOUR_LOCAL_IMAGE_DIGEST'
pytest -q tests/test_worker.py
```

Tests cover split isolation, Train-only learning, strict validation selection, immutable testing, checkpoint identity, native-tool limits, bounded IR execution, request retries and accounting, portable Bank round trips, role-equivalence proofs, transactional merging, consumer rebinding, Test-time programming, actual Docker execution, and source-backed Program qualification. `tests/test_mechanisms.py` additionally drives fabricated tasks through utility ranking, certificate and environment gating, executability downgrade, isolated validation with repair, replay repair with certification, failure localization and re-execution. Docker tests are skipped when no image digest is supplied. They are engineering checks, not performance claims.

## Repository layout

```text
src/skillcompiler/   Generic learning, execution, storage, and evaluation
configs/            Explicit runtime and resource settings
benchmark/          Portable deployment Banks
examples/           Synthetic environment and offline integration demo
tests/              Unit and container integration tests
docs/               Adapter and Bank contracts
```

## License

The framework is distributed under the [MIT License](LICENSE). Retained third-party notices for reference guidance are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
