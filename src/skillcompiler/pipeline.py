"""Train -> held-out validation -> immutable deployment -> Test."""

from dataclasses import asdict
from pathlib import Path
import random
from .bank import Bank
from .contracts import digest, read, save


class Pipeline:
    def __init__(
        self,
        suite,
        runtime,
        learner,
        output,
        batch_size=8,
        seed=42,
        refinement_rounds=1,
        min_gain=0.0,
        config_identity=None,
    ):
        self.suite, self.runtime, self.learner = suite, runtime, learner
        self.output = Path(output)
        if batch_size < 1 or refinement_rounds < 0:
            raise ValueError("Invalid training schedule")
        self.batch_size, self.seed, self.refinement_rounds, self.min_gain = (
            batch_size,
            seed,
            refinement_rounds,
            min_gain,
        )
        self.config_identity = config_identity

    def _checkpoint(self, path, bank, episodes, key, refine):
        identity = digest([bank.hash, episodes, key, refine])
        receipt_path = path / "receipt.json"
        if receipt_path.exists():
            receipt = read(receipt_path)
            candidate = Bank.load(path / "bank")
            if (
                receipt.get("identity") != identity
                or receipt.get("bank_hash") != candidate.hash
            ):
                raise ValueError("Learning checkpoint identity changed")
            return candidate
        candidate, receipt = self.learner.update(
            bank, episodes, self.suite, key, refine=refine
        )
        candidate.write(path / "bank")
        save(
            receipt_path, {**receipt, "identity": identity, "bank_hash": candidate.hash}
        )
        return candidate

    def _evaluate(self, tasks, bank, phase):
        episodes = []
        for index, task in enumerate(tasks):
            path = self.output / phase / f"{index:05d}.json"
            identity = digest([task.public(), bank.hash, phase])
            if path.exists():
                row = read(path)
                if row["identity"] != identity:
                    raise ValueError("Checkpoint inputs changed")
            else:
                environment = self.suite.open(task)
                try:
                    trace = self.runtime.run(
                        task, environment, bank, phase + "/" + str(index)
                    )
                    feedback = self.suite.score(task, trace["answer"], environment)
                    row = {**trace, "identity": identity, "feedback": asdict(feedback)}
                finally:
                    environment.close()
                save(path, row)
            episodes.append(row)
        score = (
            sum(e["feedback"]["score"] for e in episodes) / len(episodes)
            if episodes
            else 0.0
        )
        summary = {
            "tasks": len(episodes),
            "successes": sum(e["feedback"]["success"] for e in episodes),
            "score": score,
            "bank_hash": bank.hash,
        }
        save(self.output / phase / "summary.json", summary)
        return episodes, summary

    def run(self, initial_bank=None):
        splits = {s: self.suite.tasks(s) for s in ("train", "val", "test")}
        keys = {s: {t.id for t in ts} for s, ts in splits.items()}
        if any(len(keys[s]) != len(splits[s]) for s in splits):
            raise ValueError("Duplicate task IDs")
        if any(
            keys[a] & keys[b]
            for a, b in [("train", "val"), ("train", "test"), ("val", "test")]
        ):
            raise ValueError("Split overlap")
        if not all(splits.values()):
            raise ValueError("Train, validation, and Test must be nonempty")
        bank = (initial_bank or Bank()).clone()
        train = list(splits["train"])
        random.Random(self.seed).shuffle(train)
        manifest = {
            "splits": {s: digest([t.public() for t in ts]) for s, ts in splits.items()},
            "training_order": [t.id for t in train],
            "initial_bank_hash": bank.hash,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "refinement_rounds": self.refinement_rounds,
            "min_gain": self.min_gain,
        }
        manifest["configuration"] = self.config_identity
        manifest["source"] = digest(
            {
                p.name: p.read_text(encoding="utf-8")
                for p in Path(__file__).parent.glob("*.py")
            }
        )
        import inspect

        adapter_file = inspect.getsourcefile(type(self.suite))
        manifest["adapter_source"] = (
            digest(Path(adapter_file).read_text(encoding="utf-8"))
            if adapter_file
            else None
        )
        manifest["runtime"] = {
            k: getattr(self.runtime, k, None)
            for k in ("max_steps", "max_tool_calls", "retrieval_limit")
        }
        manifest["learning"] = {
            k: getattr(self.learner, k, None) for k in ("max_proposals", "repair_limit")
        }
        if (self.output / "manifest.json").exists() and read(
            self.output / "manifest.json"
        ) != manifest:
            raise ValueError("Run identity changed")
        save(self.output / "manifest.json", manifest)
        checkpoints = [bank.clone()]
        all_episodes = []
        for batch, start in enumerate(range(0, len(train), self.batch_size)):
            path = self.output / "learning" / str(batch)
            episodes, _ = self._evaluate(
                train[start : start + self.batch_size], bank, "train/" + str(batch)
            )
            all_episodes.extend(episodes)
            bank = self._checkpoint(
                path, bank, episodes, "learning/" + str(batch), batch > 0
            )
            checkpoints.append(bank.clone())
        for iteration in range(self.refinement_rounds):
            path = self.output / "refinement" / str(iteration)
            bank = self._checkpoint(
                path, bank, all_episodes, "refinement/" + str(iteration), True
            )
            checkpoints.append(bank.clone())
        unique = {b.hash: b for b in checkpoints}
        best = checkpoints[0]
        best_score = None
        validation = []
        for index, candidate in enumerate(unique.values()):
            _, summary = self._evaluate(splits["val"], candidate, "val/" + str(index))
            validation.append(summary)
            if best_score is None or summary["score"] > best_score + self.min_gain:
                best, best_score = candidate, summary["score"]
        save(
            self.output / "selection.json",
            {
                "selected_bank": best.hash,
                "candidates": validation,
                "rule": "strict score improvement beyond min_gain; ties retain earlier Bank",
            },
        )
        frozen = self.output / "frozen_bank"
        if frozen.exists() and Bank.load(frozen).hash != best.hash:
            raise ValueError("Existing frozen Bank differs")
        best.write(frozen)
        before = Bank.load(frozen).hash
        _, test = self._evaluate(splits["test"], best, "test")
        after = Bank.load(frozen).hash
        if before != after:
            raise ValueError("Test changed the frozen Bank")
        result = {
            "status": "completed",
            "train_tasks": len(train),
            "test": test,
            "frozen_hash_before": before,
            "frozen_hash_after": after,
        }
        save(self.output / "result.json", result)
        return result
