from pathlib import Path
from typing import cast
import math
from collections import Counter
from itertools import combinations
import pandas as pd

file_to_model_map = {
    "google_gen_vs_gpt5_eval": "google_gemini-2.5-pro",
    "gpt5_gen_vs_gpt5_eval": "openai_gpt-5-2025-08-07",
    "sonnet_gen_vs_gpt5_eval": "anthropic_claude-sonnet-4-5-20250929",
}

num_checks_list = [64, 54, 62]


def parse_human_review() -> dict[str, dict[str, list[str]]]:
    """Return {model_under_test: {reviewer: concatenated responses}}."""
    base_dir = Path(__file__).parent / "data" / "human_review"

    csv_frames: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(base_dir.glob("*.csv")):
        # CSVs contain a metadata header, a summary row, and a blank line before the
        # actual review table. Skip the first three rows so the detail header becomes
        # the DataFrame columns (includes "Completed By").
        csv_frames[csv_path.stem] = pd.read_csv(csv_path, skiprows=3)

    reviewer_responses: dict[str, dict[str, list[str]]] = {}
    for file_name, frame in csv_frames.items():
        start = 0
        model = file_to_model_map[file_name]
        for num_checks in num_checks_list:
            reviews = [
                frame.iloc[start + offset : start + offset + num_checks].copy()
                for offset in range(0, 3 * num_checks, num_checks)
            ]

            for review in reviews:
                reviewer = cast(str, review["Completed By"].iloc[0])
                responses = cast(list[str], review["Reviewer Response"].tolist())
                reviewer_responses.setdefault(model, {}).setdefault(
                    reviewer, []
                ).extend(responses)

            start += 3 * num_checks

    return reviewer_responses


def get_gen_and_eval_model(filename: str) -> tuple[str, str]:
    split = filename.split(" ")
    return split[0], split[3]


def parse_auto_eval() -> dict[str, dict[str, list[str]]]:
    base_dir = Path(__file__).parent / "data" / "custom_operator"

    csv_frames: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(base_dir.glob("*.csv")):
        csv_frames[csv_path.stem] = pd.read_csv(csv_path, skiprows=2)

    auto_eval_scores: dict[str, dict[str, list[str]]] = {}
    for file_name, frame in csv_frames.items():
        mutant, eval_model = get_gen_and_eval_model(file_name)
        responses = cast(list[str], frame["Auto Eval"].dropna().tolist())
        auto_eval_scores.setdefault(mutant, {})[eval_model] = responses

    return auto_eval_scores


def _validate_rating_lengths(ratings: dict[str, list[str]]) -> int:
    lengths = {len(responses) for responses in ratings.values()}
    if len(lengths) != 1:
        raise ValueError("All raters must have the same number of responses")
    return lengths.pop()


def fleiss_kappa(ratings: dict[str, list[str]]) -> float:
    """Compute Fleiss' kappa for a mapping of rater -> categorical labels."""
    if len(ratings) < 2:
        return float("nan")

    n_items = _validate_rating_lengths(ratings)
    n_raters = len(ratings)

    categories = sorted(
        {label for responses in ratings.values() for label in responses}
    )
    if not categories:
        return float("nan")
    category_index = {label: idx for idx, label in enumerate(categories)}

    counts = [[0] * len(categories) for _ in range(n_items)]
    for responses in ratings.values():
        for item_idx, label in enumerate(responses):
            counts[item_idx][category_index[label]] += 1

    item_agreements = []
    for item_counts in counts:
        numerator = sum(count * (count - 1) for count in item_counts)
        item_agreements.append(numerator / (n_raters * (n_raters - 1)))

    p_bar = sum(item_agreements) / n_items

    category_totals = [0] * len(categories)
    for item_counts in counts:
        for idx, count in enumerate(item_counts):
            category_totals[idx] += count
    total_ratings = n_items * n_raters
    category_props = [total / total_ratings for total in category_totals]
    p_e = sum(prop * prop for prop in category_props)

    denom = 1 - p_e
    if math.isclose(denom, 0.0):
        return 1.0 if math.isclose(p_bar, 1.0) else float("nan")

    return (p_bar - p_e) / denom


def gather_ratings_for_mutation(
    human_review: dict[str, dict[str, list[str]]],
    auto_eval: dict[str, dict[str, list[str]]],
    mutant: str,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    humans = human_review.get(mutant, {})
    models = auto_eval.get(mutant, {})
    return dict(humans), dict(models)


def compute_human_majority(
    human_ratings: dict[str, list[str]],
) -> list[str | None]:
    if not human_ratings:
        return []
    human_raters = list(human_ratings.keys())
    n_items = len(next(iter(human_ratings.values())))
    majority: list[str | None] = []
    for idx in range(n_items):
        labels = [human_ratings[r][idx] for r in human_raters]
        counts = Counter(labels)
        if not counts:
            majority.append(None)
            continue
        best, best_count = counts.most_common(1)[0]
        # check for ties
        tied = [label for label, count in counts.items() if count == best_count]
        if len(tied) > 1:
            majority.append(None)
        else:
            majority.append(best)
    return majority


def agreement_percentage(
    human_majority: list[str | None],
    model_labels: list[str],
) -> float:
    comparisons = [
        (h, m) for h, m in zip(human_majority, model_labels) if h is not None
    ]
    if not comparisons:
        return float("nan")
    matches = sum(1 for h, m in comparisons if h == m)
    return 100.0 * matches / len(comparisons)


def pairwise_agreement(ratings: dict[str, list[str]]) -> float:
    if len(ratings) < 2:
        return float("nan")
    n_items = len(next(iter(ratings.values())))
    agreeing_pairs = 0
    total_pairs = 0
    raters = list(ratings.keys())
    for idx in range(n_items):
        labels = {r: ratings[r][idx] for r in raters}
        for first, second in combinations(raters, 2):
            total_pairs += 1
            if labels[first] == labels[second]:
                agreeing_pairs += 1
    if total_pairs == 0:
        return float("nan")
    return 100.0 * agreeing_pairs / total_pairs


if __name__ == "__main__":
    human_review = parse_human_review()
    auto_eval = parse_auto_eval()

    mutants = sorted(set(human_review.keys()) | set(auto_eval.keys()))
    for mutant in mutants:
        human_ratings, model_ratings = gather_ratings_for_mutation(
            human_review, auto_eval, mutant
        )
        if len(human_ratings) < 2:
            print(f"{mutant}: insufficient human raters ({len(human_ratings)})")
            continue

        human_pairwise = pairwise_agreement(human_ratings)
        majority = compute_human_majority(human_ratings)
        auto_agreements = []
        for model_name, labels in model_ratings.items():
            pct = agreement_percentage(majority, labels)
            auto_agreements.append(f"{model_name} vs majority={pct:.2f}%")
        auto_str = ", ".join(auto_agreements) if auto_agreements else "no auto eval"
        print(
            f"{mutant}: human raters={len(human_ratings)} pairwise agreement={human_pairwise:.2f}%, "
            f"{auto_str}"
        )
