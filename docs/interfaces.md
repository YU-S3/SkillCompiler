# Adapter and execution interfaces

## Public task boundary

`Task(id, instruction, inputs)` is the complete public task view. IDs identify records; they must not encode the answer. Splits use distinct IDs. The adapter owns private labels and scoring data. `Feedback(score, success, detail)` is collected after execution; only Train feedback is sent to the learner.

`open(task)` must create a reset environment. `observe()` returns public state. `tools()` returns objects containing `name`, `description`, `input_schema`, and optionally `output_schema`. `call(name, arguments)` performs one public native operation. `close()` releases resources. The core validates calls and records their actual results.

Images, files, and large results must be exposed through your adapter's public tools or supported message representation; the bundled chat transport currently sends JSON text. It does not implement a multimodal upload protocol. A file tool should restrict access to the task workspace and publish only validated relative paths.

## Program ABI

```python
def run(ctx, inputs):
    result = ctx.call("public_tool", {"argument": inputs["argument"]})
    return {"status": "ok", "outputs": {"result": result}}
```

Allowed statuses are `ok`, `not_found`, `needs_input`, and `blocked`. Outputs must be an object; `ok` outputs must satisfy the declared schema. Native calls are limited by both Program permissions and the task tool budget. Programs receive no private evaluator state.

`ctx.observe()`, `ctx.available_tools()`, and `ctx.remaining_calls()` expose the public environment. An optional `ctx.read_result(...)` delegates to the adapter's public result reader and consumes one tool call. For a named `read_result` native tool, use `ctx.call()` and declare that tool in `allowed_tools`.

The direct `ctx.read_result(...)` form also requires `read_result` in `allowed_tools`.

For semantic contracts, adapters can implement `verify_preconditions(conditions, inputs)`, `verify_effects(effects, inputs, variables, result)`, and `verify_contract(validator, inputs, result)`. They must verify public execution facts without consulting gold answers. When an asset declares one of these checks, a missing verifier or a return value other than `True` blocks admission/execution. JSON input validation alone cannot establish semantic prerequisites.

## Qualification

A proposed Program names replay cases from the current training material. Each case specifies:

```json
{
  "episode": "training-record-id",
  "inputs": {"argument": "public value"},
  "prefix": [0, 1],
  "expected_from": {"event": 2, "path": ["result"]}
}
```

The Host resolves the expected object from an executed observation, resets the environment, replays the declared native prefix, and executes the Program in Docker. It compares the actual output with that observed object. Model-supplied expected answers are not accepted. Prefix indices must precede the reference event.

Passing replay establishes compatibility with those observations, not general correctness. Held-out validation selects the final Bank. The framework does not automatically confer universal correctness or a permanent success label on a Program.

Runtime can also request a new task-local Program during Train, Val or Test. This path records structural admission and actual bounded execution separately, without inserting the result into a persistent Bank. See [semantic alignment and Runtime programming](semantic-alignment.md) for the complete interface and merge gates.

## Model boundary

The model object implements `ask(key, prompt, material, schema) -> dict`. The supplied `ChatModel` sends the schema with the request and validates the parsed result locally. A stable key identifies a logical request. Physical attempts receive separate durable receipts and budget entries. Another provider can implement this same interface without changing the learner or pipeline.

The run configuration is not a place for API keys: store only an environment-variable name. Endpoint/model/generation options are included in request identities. Keep credentials in the process environment.
