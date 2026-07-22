from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import re
import argparse


def context_length_tag(context_length):
    if context_length % 1024 == 0:
        return f"{context_length // 1024}k"
    return str(context_length)


parser = argparse.ArgumentParser(
    description="Analyze RULER experiment results"
)

parser.add_argument(
    "--context-length",
    type=int,
    required=True,
)

parser.add_argument(
    "--tasks",
    nargs="+",
    required=True,
)

parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
)

args = parser.parse_args()


MODEL = "Llama-3-8B-Instruct-Gradient-1048k"
CONTEXT_LENGTH = str(args.context_length)
LENGTH_TAG = context_length_tag(args.context_length)
TASKS = args.tasks

MODES = [
    "Full_Flash_Attn",
    "RetroInfer",
    "SpecDecoder",
]

ROOT = (
    Path("ruler_eval_result")
    / "gradientai"
    / MODEL
    / "synthetic"
    / CONTEXT_LENGTH
)

OUTPUT_DIR = (
    args.output_dir
    if args.output_dir is not None
    else Path(f"analysis_{LENGTH_TAG}")
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def load_score(path):
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    score = None
    nulls = None

    for row in rows:
        if "Score" in row:
            position = row.index("Score")
            score = float(row[position + 1])

        if "Nulls" in row:
            position = row.index("Nulls")
            nulls = row[position + 1]

    if score is None:
        raise ValueError(f"Score not found in {path}")

    return score, nulls


def file_sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def common_prefix_length(first, second):
    length = min(len(first), len(second))

    for position in range(length):
        if first[position] != second[position]:
            return position

    return length


def reference_score(task, prediction, references):
    prediction = prediction.lower()
    references = [reference.lower() for reference in references]

    if task.startswith("qa_"):
        return 100.0 if any(
            reference in prediction for reference in references
        ) else 0.0

    matched = sum(
        reference in prediction for reference in references
    )

    return 100.0 * matched / len(references)


task_rows = []
sample_rows = []

for task in TASKS:
    full_prediction_path = (
        ROOT
        / "Full_Flash_Attn"
        / task
        / "pred"
        / f"{task}.jsonl"
    )

    full_data_path = (
        ROOT
        / "Full_Flash_Attn"
        / task
        / "data"
        / task
        / "validation.jsonl"
    )

    full_predictions = {
        row["index"]: row
        for row in load_jsonl(full_prediction_path)
    }

    full_data = {
        row["index"]: row
        for row in load_jsonl(full_data_path)
    }

    full_input_hash = file_sha256(full_data_path)

    full_score, _ = load_score(
        ROOT
        / "Full_Flash_Attn"
        / task
        / "pred"
        / f"summary-{task}.csv"
    )

    for mode in MODES:
        prediction_path = (
            ROOT
            / mode
            / task
            / "pred"
            / f"{task}.jsonl"
        )

        data_path = (
            ROOT
            / mode
            / task
            / "data"
            / task
            / "validation.jsonl"
        )

        summary_path = (
            ROOT
            / mode
            / task
            / "pred"
            / f"summary-{task}.csv"
        )

        predictions = {
            row["index"]: row
            for row in load_jsonl(prediction_path)
        }

        score, nulls = load_score(summary_path)
        input_matches_full = (
            file_sha256(data_path) == full_input_hash
        )

        indices = sorted(full_predictions)
        exact_count = 0
        prefix_ratios = []

        for index in indices:
            full_ids = full_predictions[index]["token_ids"]
            current_ids = predictions[index]["token_ids"]

            exact = full_ids == current_ids
            exact_count += int(exact)

            prefix_length = common_prefix_length(
                full_ids,
                current_ids,
            )

            denominator = max(
                len(full_ids),
                len(current_ids),
                1,
            )

            prefix_ratio = prefix_length / denominator
            prefix_ratios.append(prefix_ratio)

            references = full_data[index]["outputs"]
            prediction = predictions[index]["pred"]

            sample_rows.append({
                "task": task,
                "index": index,
                "mode": mode,
                "reference_score": round(
                    reference_score(
                        task,
                        prediction,
                        references,
                    ),
                    4,
                ),
                "exact_to_full": exact,
                "first_diff_position": (
                    -1 if exact else prefix_length
                ),
                "common_prefix_length": prefix_length,
                "common_prefix_ratio": round(
                    prefix_ratio,
                    6,
                ),
                "generated_tokens": len(current_ids),
                "prediction": prediction.replace("\n", "\\n"),
            })

        sample_count = len(indices)

        task_rows.append({
            "task": task,
            "mode": mode,
            "samples": sample_count,
            "score": score,
            "score_delta_vs_full": round(
                score - full_score,
                4,
            ),
            "exact_samples_vs_full": exact_count,
            "exact_sample_rate_vs_full": round(
                100.0 * exact_count / sample_count,
                2,
            ),
            "mean_common_prefix_ratio": round(
                sum(prefix_ratios) / len(prefix_ratios),
                6,
            ),
            "input_matches_full": input_matches_full,
            "nulls": nulls,
        })


def write_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


write_csv(
    OUTPUT_DIR / f"task_summary_{LENGTH_TAG}.csv",
    task_rows,
)

write_csv(
    OUTPUT_DIR / f"sample_comparison_{LENGTH_TAG}.csv",
    sample_rows,
)


# SpecDecoder机制统计
summary_pattern = re.compile(
    r"Draft tokens: (\d+), "
    r"Sparse accept tokens: (\d+), "
    r"Sparse reject tokens: (\d+), "
    r"Full accept tokens: (\d+), "
    r"Full reject tokens: (\d+), "
    r"Generate steps: (\d+)"
)

draft_stop_pattern = re.compile(
    r"Draft stopped by ([^\s]+)"
)

full_trigger_pattern = re.compile(
    r"Full verify by ([^:]+):"
)

mechanism_rows = []

for task in TASKS:
    log_path = Path(
        f"logs/{task}_{LENGTH_TAG}_SpecDecoder.log"
    )

    text = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    summaries = [
        tuple(map(int, match.groups()))
        for match in summary_pattern.finditer(text)
    ]

    if not summaries:
        raise ValueError(
            f"No SpecDecoder summary found in {log_path}"
        )

    draft_tokens = sum(row[0] for row in summaries)
    sparse_accept = sum(row[1] for row in summaries)
    sparse_reject = sum(row[2] for row in summaries)
    full_accept = sum(row[3] for row in summaries)
    full_reject = sum(row[4] for row in summaries)
    generate_steps = sum(row[5] for row in summaries)

    sparse_verified = sparse_accept + sparse_reject
    full_verified = full_accept + full_reject

    draft_stop_reasons = Counter(
        draft_stop_pattern.findall(text)
    )

    full_trigger_reasons = Counter(
        full_trigger_pattern.findall(text)
    )

    mechanism_rows.append({
        "task": task,
        "samples": len(summaries),
        "draft_tokens": draft_tokens,
        "sparse_verified_tokens": sparse_verified,
        "sparse_accept_tokens": sparse_accept,
        "sparse_reject_tokens": sparse_reject,
        "discarded_draft_suffix_tokens": (
            draft_tokens - sparse_verified
        ),
        "draft_utilization_rate": round(
            sparse_accept / draft_tokens,
            6,
        ) if draft_tokens else 0.0,
        "draft_to_sparse_accept_rate": round(
            sparse_accept / sparse_verified,
            6,
        ) if sparse_verified else 0.0,
        "full_verified_tokens": full_verified,
        "full_accept_tokens": full_accept,
        "full_reject_tokens": full_reject,
        "sparse_to_full_accept_rate": round(
            full_accept / full_verified,
            6,
        ) if full_verified else 0.0,
        "generate_steps": generate_steps,
        "full_verify_deferred": text.count(
            "Full verify deferred"
        ),
        "draft_stop_reasons": json.dumps(
            draft_stop_reasons,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "full_trigger_reasons": json.dumps(
            full_trigger_reasons,
            ensure_ascii=False,
            sort_keys=True,
        ),
    })


write_csv(
    OUTPUT_DIR / f"mechanism_summary_{LENGTH_TAG}.csv",
    mechanism_rows,
)


print("\nTask-level results")
print(
    "| Task | Mode | Score | ΔFull | Exact/Full | Prefix | Same input |"
)
print(
    "|---|---|---:|---:|---:|---:|---|"
)

for row in task_rows:
    print(
        f"| {row['task']} "
        f"| {row['mode']} "
        f"| {row['score']:.2f} "
        f"| {row['score_delta_vs_full']:+.2f} "
        f"| {row['exact_sample_rate_vs_full']:.2f}% "
        f"| {row['mean_common_prefix_ratio']:.4f} "
        f"| {row['input_matches_full']} |"
    )


print("\nSpecDecoder mechanism results")
print(
    "| Task | Draft | DS accept | SF accept | Deferred | Steps |"
)
print(
    "|---|---:|---:|---:|---:|---:|"
)

for row in mechanism_rows:
    print(
        f"| {row['task']} "
        f"| {row['draft_tokens']} "
        f"| {100 * row['draft_to_sparse_accept_rate']:.2f}% "
        f"| {100 * row['sparse_to_full_accept_rate']:.2f}% "
        f"| {row['full_verify_deferred']} "
        f"| {row['generate_steps']} |"
    )


print(f"\nSaved results to: {OUTPUT_DIR.resolve()}")