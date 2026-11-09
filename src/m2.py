import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from prototype import metrics
from prototype.label_map import map_fine_grained

COMMIT = "a0c55a945497ab58e94aef491180e1cf88ffb864"
HASHES = {
    "train.csv": "16ca7955bdcaf90f5d8ac39e51eb3eda15320caa1e730ed45cb5e8ca8cd0d5e8",
    "dev.csv": "28e67a26815aa1852449ba06919e51c07060d146e2bd23101cc0137a0ff94b7c",
    "test.csv": "b06d31481f0d558b6dc8140408938abe8dba8083cd9976b4ba8f5c837fbe78ed",
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
}
SEED = 20261109
ALIASES = {"0": 0, "safe": 0, "正常": 0, "1": 1, "offensive": 1, "冒犯": 1}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_rows(raw_dir, split):
    name = f"{split}.csv"
    path = raw_dir / name
    if digest(path.read_bytes()) != HASHES[name]:
        raise ValueError(f"{name}: SHA-256 mismatch")
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as source:
        for number, row in enumerate(csv.DictReader(source), 1):
            text = row["TEXT"].strip()
            label = ALIASES[row["label"]]
            fine = row["fine-grained-label"] if split == "test" else None
            if not text or (fine is not None and map_fine_grained(fine) != label):
                raise ValueError(f"{name}: invalid record {number}")
            rows.append({"id": f"{split}-{number:05d}", "text": text, "label": label,
                         "topic": row["topic"], "fine": fine})
    return rows


def clean(rows, excluded):
    labels = defaultdict(set)
    for row in rows:
        labels[row["text"]].add(row["label"])
    seen = set()
    kept = []
    drops = Counter()
    for row in rows:
        key = row["text"]
        if key in excluded:
            drops["cross_split"] += 1
        elif len(labels[key]) > 1:
            drops["conflicting_label"] += 1
        elif key in seen:
            drops["within_split_repeat"] += 1
        else:
            seen.add(key)
            kept.append(row)
    return kept, dict(drops)


def text_keys(rows):
    return {row["text"] for row in rows}


def stats(rows):
    return {"n": len(rows), "normal": sum(row["label"] == 0 for row in rows),
            "offensive": sum(row["label"] == 1 for row in rows),
            "avg_trimmed_chars": round(sum(len(row["text"]) for row in rows) / len(rows), 2),
            "topic": dict(Counter(row["topic"] for row in rows))}


def test_subset(rows):
    selected = []
    for label in (0, 1):
        group = (row["id"] for row in rows if row["label"] == label)
        ordered = sorted(group, key=lambda item: (digest(f"{SEED}:{item}".encode("ascii")), item))
        selected.extend(ordered[:50])
    return sorted(selected)


def write_ids(path, ids):
    path.write_bytes(("id\n" + "\n".join(ids) + "\n").encode("ascii"))


def prepare(raw_dir, out, emit_data=False):
    if digest((raw_dir / "LICENSE").read_bytes()) != HASHES["LICENSE"]:
        raise ValueError("LICENSE: SHA-256 mismatch")
    raw = {name: read_rows(raw_dir, name) for name in ("train", "dev", "test")}
    test = raw["test"]
    dev, dev_drops = clean(raw["dev"], text_keys(test))
    train, train_drops = clean(raw["train"], text_keys(test) | text_keys(raw["dev"]))
    assert not (text_keys(train) & text_keys(dev) or text_keys(train) & text_keys(test)
                or text_keys(dev) & text_keys(test))
    files = {}
    for group in ("development", "acceptance"):
        (out / group).mkdir(parents=True, exist_ok=True)
    splits = {"development/train": train, "development/dev": dev, "acceptance/test": test}
    for name, rows in splits.items():
        files[f"{name}_ids.csv"] = [row["id"] for row in rows]
        if emit_data:
            data = rows if name.startswith("development") else [
                {"id": row["id"], "text": row["text"]} for row in rows]
            with (out / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as target:
                for row in data:
                    target.write(json.dumps(row, ensure_ascii=False) + "\n")
    files["acceptance/common100_ids.csv"] = test_subset(test)
    for name, ids in files.items():
        write_ids(out / name, ids)
    combined = raw["train"] + raw["dev"] + test
    info = {
        "version": "M2-v1.0",
        "freeze_date": "2026-11-09",
        "source": "https://github.com/thu-coai/COLDataset/tree/" + COMMIT,
        "source_sha256": HASHES,
        "implementation_revision": "r2",
        "scorer_and_splitter_sha256": digest(Path(__file__).read_bytes()),
        "inherited_module_sha256": {
            f"prototype/{name}": digest(Path(__file__).with_name("prototype").joinpath(name).read_bytes())
            for name in ("label_map.py", "metrics.py")
        },
        "id_file_sha256": {name: digest((out / name).read_bytes()) for name in files},
        "raw": {name: stats(rows) for name, rows in raw.items()},
        "raw_total": stats(combined),
        "retained": {name: stats(rows) for name, rows in splits.items()},
        "dropped": {"train": train_drops, "dev": dev_drops},
        "raw_exact_overlap": {
            "train_dev": len(text_keys(raw["train"]) & text_keys(raw["dev"])),
            "train_test": len(text_keys(raw["train"]) & text_keys(test)),
            "dev_test": len(text_keys(raw["dev"]) & text_keys(test)),
        },
        "test_internal_repeats": len(test) - len(text_keys(test)),
        "test_fine": dict(Counter(row["fine"] for row in test)),
        "test_by_topic": {
            topic: {"normal": sum(row["topic"] == topic and row["label"] == 0 for row in test),
                    "offensive": sum(row["topic"] == topic and row["label"] == 1 for row in test)}
            for topic in sorted({row["topic"] for row in test})
        },
        "common100": {"seed": SEED, "normal": 50, "offensive": 50},
        "all_safe_reference": evaluate(test, {row["id"]: 0 for row in test}),
        "acceptance_thresholds": {"macro_f1": 0.8, "offensive_recall": 0.8,
                                  "agent_macro_f1_gain_pp": 3, "agent_antibias_error_drop_pp": 5},
    }
    (out / "freeze.json").write_bytes((json.dumps(info, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return info


def normalize_label(item):
    if type(item) is int and item in (0, 1):
        return item
    if isinstance(item, str):
        return ALIASES.get(item.strip().lower())
    return None


def read_predictions(path, valid_ids):
    predictions = {}
    with path.open(encoding="utf-8-sig") as source:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            key = item["id"]
            if key not in valid_ids or key in predictions:
                raise ValueError(f"Unknown or repeated prediction ID: {key}")
            predictions[key] = normalize_label(item.get("label")) if item.get("status") == "ok" else None
    return predictions


def evaluate(rows, predictions):
    truth = [row["label"] for row in rows]
    predicted = [predictions.get(row["id"]) for row in rows]
    tp, fp, fn, tn = metrics.confusion(truth, predicted)
    counts = Counter(zip(truth, predicted))
    for row, prediction in zip(rows, predicted):
        if row["fine"] == "3":
            counts["anti", prediction] += 1

    invalid_normal, invalid_offensive = counts[0, None], counts[1, None]
    anti = sum(counts["anti", value] for value in (0, 1, None))

    def ratio(num, denom):
        return num / denom if denom else 0.0

    macro = (metrics.f1_score(tn, fn, fp) + metrics.f1_score(tp, fp, fn)) / 2
    return {
        "n": len(rows), "macro_f1": macro, "accuracy": metrics.accuracy(truth, predicted),
        "valid_rate": metrics.valid_rate(predicted),
        "offensive_recall": ratio(tp, tp + fn),
        "normal_error_rate": ratio(fp, fp + tn),
        "normal_explicit_fp_rate": ratio(counts[0, 1], fp + tn),
        "normal_invalid": invalid_normal, "offensive_invalid": invalid_offensive,
        "anti_bias_n": anti,
        "anti_bias_error_rate": ratio(counts["anti", 1] + counts["anti", None], anti),
        "anti_bias_explicit_fp_rate": ratio(counts["anti", 1], anti),
        "anti_bias_invalid": counts["anti", None],
        "penalized_confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def selftest():
    rows = [{"id": "a", "label": 0, "fine": "3"}, {"id": "b", "label": 1, "fine": "1"},
            {"id": "c", "label": 0, "fine": "0"}, {"id": "d", "label": 1, "fine": "2"}]
    result = evaluate(rows, {"a": None, "c": 0, "d": 1})
    assert result["macro_f1"] == result["accuracy"] == result["valid_rate"] == 0.5
    assert result["offensive_recall"] == result["normal_error_rate"] == 0.5
    assert result["anti_bias_error_rate"] == 1 and result["anti_bias_explicit_fp_rate"] == 0
    assert normalize_label(" SAFE ") == 0 and normalize_label("冒犯") == 1
    assert all(normalize_label(value) is None for value in (True, 1.0, "0/1", "标签:1", 2, []))
    assert evaluate(rows, {})["macro_f1"] == 0
    print("selftest passed")


def main():
    parser = argparse.ArgumentParser(description="COLD M2-v1.0")
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_cmd = sub.add_parser("prepare")
    prepare_cmd.add_argument("raw_dir", type=Path)
    prepare_cmd.add_argument("output_dir", type=Path)
    prepare_cmd.add_argument("--emit-data", action="store_true")
    score = sub.add_parser("score")
    score.add_argument("raw_dir", type=Path)
    score.add_argument("predictions", type=Path)
    score.add_argument("--common100", action="store_true")
    sub.add_parser("selftest")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.raw_dir, args.output_dir, args.emit_data)
    elif args.action == "score":
        rows = read_rows(args.raw_dir, "test")
        if args.common100:
            selected = set(test_subset(rows))
            rows = [row for row in rows if row["id"] in selected]
        predictions = read_predictions(args.predictions, {row["id"] for row in rows})
        print(json.dumps(evaluate(rows, predictions), ensure_ascii=False, indent=2))
    else:
        selftest()


if __name__ == "__main__":
    main()

