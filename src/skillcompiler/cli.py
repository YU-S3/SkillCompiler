import argparse
import importlib
import importlib.util
import sys
import json
from pathlib import Path
import yaml
from .bank import Bank
from .model import ChatModel, Ledger
from .runtime import Runtime
from .learning import Learner
from .pipeline import Pipeline
from .worker import ProgramWorker
from .contracts import digest, save
from .locking import run_lock


def main():
    parser = argparse.ArgumentParser(prog="skillcompiler")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("check-bank")
    inspect.add_argument("path")
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--output", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--bank", required=True)
    evaluate.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()
    if args.command == "check-bank":
        bank = Bank.load(args.path)
        print(
            json.dumps(
                {"hash": bank.hash, **{k: len(v) for k, v in bank.data().items()}},
                indent=2,
            )
        )
        return
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.output)
    module, name = cfg["adapter"]["factory"].split(":", 1)
    if module.endswith(".py"):
        path = (Path(args.config).resolve().parent / module).resolve()
        spec = importlib.util.spec_from_file_location(
            "skillcompiler_external_adapter", path
        )
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = loaded
        spec.loader.exec_module(loaded)
    else:
        loaded = importlib.import_module(module)
    suite = getattr(loaded, name)(**cfg["adapter"].get("parameters", {}))
    with run_lock(root):
        ledger = Ledger(root / "usage.json", **cfg["budget"])
        model = ChatModel(cfg["model"], root / "requests", ledger)
        worker = ProgramWorker(**cfg["worker"]) if cfg.get("worker") else None
        runtime = Runtime(model, worker, **cfg.get("runtime", {}))
        learner = Learner(model, worker, **cfg.get("learning", {}))
        pipeline = Pipeline(
            suite,
            runtime,
            learner,
            root,
            config_identity=digest(cfg),
            **cfg.get("training", {}),
        )
        if args.command == "evaluate":
            from .contracts import read

            bank = Bank.load(args.bank)
            before = bank.hash
            identity = {
                "config_hash": digest(cfg),
                "bank_hash": before,
                "split": args.split,
            }
            if (root / "evaluation.json").exists() and read(
                root / "evaluation.json"
            ) != identity:
                raise ValueError("Evaluation identity changed")
            save(root / "evaluation.json", identity)
            _, summary = pipeline._evaluate(suite.tasks(args.split), bank, args.split)
            after = Bank.load(args.bank).hash
            if before != after:
                raise ValueError("Evaluation modified the Bank")
            result = {
                "status": "completed",
                "summary": summary,
                "frozen_hash_before": before,
                "frozen_hash_after": after,
            }
        else:
            initial = (
                Bank.load(cfg["initial_bank"]) if cfg.get("initial_bank") else None
            )
            result = pipeline.run(initial)
        result["usage"] = ledger.summary()
        save(root / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
