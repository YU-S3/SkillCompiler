"""Fabricated two-tool inventory tasks for a live end-to-end check (not a benchmark)."""

from skillcompiler.contracts import Task, Feedback

PRICES = {"widget": 7, "gadget": 12, "sprocket": 5, "flange": 9, "gizmo": 4}
STOCK = {"widget": 6, "gadget": 3, "sprocket": 11, "flange": 8, "gizmo": 13}


def schema(role, kind):
    return {
        "type": "object",
        "properties": {role: {"type": kind}},
        "required": [role],
        "additionalProperties": False,
    }


class Environment:
    def __init__(self, task):
        self.task = task
        self.price, self.stock = {}, {}

    def observe(self):
        return {"catalog": sorted(PRICES)}

    def tools(self):
        return [
            {
                "name": "get_price",
                "description": "Unit price of an item in whole dollars",
                "input_schema": schema("item", "string"),
                "output_schema": schema("price", "integer"),
            },
            {
                "name": "get_stock",
                "description": "Units of an item currently in stock",
                "input_schema": schema("item", "string"),
                "output_schema": schema("stock", "integer"),
            },
        ]

    def call(self, name, arguments):
        item = arguments["item"]
        if item not in PRICES:
            raise ValueError("Unknown item")
        if name == "get_price":
            self.price[item] = PRICES[item]
            return {"price": PRICES[item]}
        if name == "get_stock":
            self.stock[item] = STOCK[item]
            return {"stock": STOCK[item]}
        raise ValueError("Unknown tool")

    def predicate_contracts(self):
        return [
            {
                "predicate": "observed_price",
                "effect_domain": "evidence",
                "args": {"item": "item name", "price": "observed unit price"},
                "description": "get_price returned this price for this item.",
            },
            {
                "predicate": "observed_stock",
                "effect_domain": "evidence",
                "args": {"item": "item name", "stock": "observed units in stock"},
                "description": "get_stock returned this stock for this item.",
            },
            {
                "predicate": "inventory_value",
                "effect_domain": "evidence",
                "args": {"item": "item name", "value": "price times stock"},
                "description": "value equals the observed price times the observed stock of item.",
            },
        ]

    def verify_effects(self, effects, inputs, variables, result):
        def value(ref):
            if ref.get("kind") in {"skill_input", "tool_input"}:
                return inputs[ref["source_role"]]
            if ref.get("kind") in {"skill_output", "tool_output"}:
                return result["outputs"][ref["source_role"]]
            if ref.get("kind") == "constant":
                return ref["value"]
            raise ValueError("Unsupported predicate role")

        try:
            for effect in effects:
                if (
                    effect.get("effect_domain", "world") != "evidence"
                    or effect.get("cardinality", 1) != 1
                    or effect.get("distinct_by", "")
                ):
                    return False
                args = {k: value(v) for k, v in effect["args"].items()}
                item = args.get("item")
                name = effect["predicate"]
                if name == "observed_price":
                    ok = item in self.price and args.get("price") == self.price[item]
                elif name == "observed_stock":
                    ok = item in self.stock and args.get("stock") == self.stock[item]
                elif name == "inventory_value":
                    ok = (
                        item in self.price
                        and item in self.stock
                        and args.get("value") == self.price[item] * self.stock[item]
                    )
                else:
                    ok = False
                if not ok:
                    return False
        except (KeyError, TypeError):
            return False
        return True

    def verify_preconditions(self, conditions, inputs):
        return self.verify_effects(conditions, inputs, {}, {"outputs": {}})

    def close(self):
        pass


class Suite:
    SPLITS = {
        "train": ["widget", "gadget"],
        "val": ["sprocket"],
        "test": ["flange", "gizmo"],
    }

    def tasks(self, split):
        return [
            Task(
                split + "_" + item,
                "Compute the total inventory value of the item (unit price times units in stock) and return it as a JSON integer, without an enclosing object.",
                {"item": item},
            )
            for item in self.SPLITS[split]
        ]

    def open(self, task):
        return Environment(task)

    def score(self, task, answer, environment):
        item = task.inputs["item"]
        success = answer == PRICES[item] * STOCK[item]
        return Feedback(float(success), success)
