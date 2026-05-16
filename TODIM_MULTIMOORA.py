"""
TODIM-MULTMOOA Evaluation System for All Learners.
Original author: CTBU_XuBJ
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


LinguisticTuple = tuple[str, float]
DecisionMatrix = list[list[float]]
RankedResults = list[tuple[str, float]]

OUTPUT_DIR = Path("evaluation_results")
RESULT_FILE_PREFIX = "TODIM_MULTMOOA_Results"

LINGUISTIC_TERM_SET: dict[str, int] = {
    "s_0": 0,
    "s_1": 1,
    "s_2": 2,
    "s_3": 3,
    "s_4": 4,
    "s_5": 5,
    "s_6": 6,
}

# Weight vector for TODIM criteria.
CRITERIA_WEIGHTS = [1.0, 0.2233, 0.4433, 0.6667, 0.3333, 0.6667]

# TODIM loss attenuation coefficient.
LOSS_ATTENUATION_FACTOR = 2.0

# Comprehensive weight vector used by MULTIMOORA methods.
COMPREHENSIVE_WEIGHTS = [0.3, 0.067, 0.133, 0.2, 0.1, 0.2]

# All criteria are benefit criteria in the original script.
BENEFIT_CRITERIA_COUNT = len(COMPREHENSIVE_WEIGHTS)

RAW_DECISION_MATRIX: list[list[LinguisticTuple]] = [
    [("s_3", -0.28), ("s_0", 0.23), ("s_2", 0.27), ("s_5", 0.44), ("s_3", 0.45), ("s_5", 0.28)],
    [("s_2", 0.31), ("s_0", 0.43), ("s_4", 0.37), ("s_5", -0.43), ("s_6", 0.0), ("s_5", 0.39)],
    [("s_5", 0.25), ("s_4", 0.42), ("s_6", 0.0), ("s_6", 0.0), ("s_0", 0.44), ("s_4", 0.29)],
    [("s_5", 0.21), ("s_1", 0.22), ("s_3", 0.18), ("s_0", 0.29), ("s_1", 0.13), ("s_2", 0.19)],
    [("s_1", 0.48), ("s_5", -0.16), ("s_5", 0.34), ("s_0", 0.22), ("s_2", 0.27), ("s_4", 0.19)],
    [("s_4", 0.15), ("s_0", 0.38), ("s_2", 0.27), ("s_4", -0.24), ("s_1", 0.18), ("s_0", 0.43)],
    [("s_1", 0.31), ("s_5", 0.37), ("s_0", 0.4), ("s_6", 0.0), ("s_2", 0.32), ("s_1", 0.45)],
    [("s_6", 0.0), ("s_4", -0.25), ("s_4", 0.46), ("s_4", 0.44), ("s_6", 0.0), ("s_3", 0.25)],
    [("s_6", 0.0), ("s_1", 0.21), ("s_5", 0.2), ("s_5", 0.24), ("s_0", 0.43), ("s_0", 0.41)],
    [("s_2", 0.44), ("s_0", 0.26), ("s_6", 0.0), ("s_1", 0.39), ("s_3", 0.31), ("s_1", 0.28)],
    [("s_0", 0.44), ("s_2", 0.12), ("s_3", 0.4), ("s_2", 0.47), ("s_3", 0.49), ("s_6", 0.0)],
    [("s_4", 0.38), ("s_1", 0.21), ("s_5", -0.28), ("s_4", 0.43), ("s_3", 0.19), ("s_3", 0.46)],
    [("s_6", 0.0), ("s_1", 0.22), ("s_2", 0.2), ("s_5", 0.4), ("s_6", 0.0), ("s_0", 0.16)],
    [("s_4", 0.44), ("s_1", 0.34), ("s_1", 0.28), ("s_0", 0.23), ("s_5", 0.3), ("s_1", 0.2)],
    [("s_3", 0.29), ("s_0", 0.43), ("s_2", 0.23), ("s_6", 0.0), ("s_5", 0.47), ("s_5", 0.46)],
    [("s_6", 0.0), ("s_2", 0.26), ("s_1", 0.22), ("s_3", 0.3), ("s_4", 0.5), ("s_1", 0.21)],
    [("s_5", 0.39), ("s_4", 0.46), ("s_2", 0.46), ("s_0", 0.33), ("s_3", 0.17), ("s_6", 0.0)],
    [("s_5", 0.19), ("s_1", 0.24), ("s_5", 0.3), ("s_1", 0.25), ("s_1", 0.14), ("s_2", 0.4)],
    [("s_5", 0.23), ("s_5", 0.31), ("s_6", 0.0), ("s_3", 0.44), ("s_0", 0.31), ("s_2", 0.42)],
    [("s_0", 0.21), ("s_1", 0.38), ("s_0", 0.3), ("s_0", 0.39), ("s_1", 0.12), ("s_1", 0.3)],
]


class ResultLogger:
    """Print results to the console and persist them to a result file."""

    def __init__(self, output_dir: Path = OUTPUT_DIR) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = output_dir / f"{RESULT_FILE_PREFIX}_{timestamp}.txt"

    def write(self, content: object = "", title: str | None = None) -> None:
        """Print content and append it to the result file."""
        lines: list[str] = []

        if title:
            lines.append(f"\n=== {title} ===")
        lines.append(str(content))

        for line in lines:
            print(line)

        with self.file_path.open("a", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")


def learner_name(index: int) -> str:
    """Return the 1-based learner label used in reports."""
    return f"Stud{index + 1}"


def calculate_numerical_scale(
    linguistic_terms: dict[str, int],
    term: str,
    alpha: float,
) -> float:
    """Convert a 2-tuple linguistic value into a numerical scale value."""
    current_index = int(term.split("_")[1])
    current_value = linguistic_terms[term]

    if alpha >= 0 and f"s_{current_index + 1}" in linguistic_terms:
        next_value = linguistic_terms[f"s_{current_index + 1}"]
        return current_value + alpha * (next_value - current_value)

    previous_key = f"s_{current_index - 1}"
    if alpha < 0 and previous_key not in linguistic_terms:
        raise ValueError(f"Cannot apply negative alpha {alpha} to the first term {term}.")

    previous_value = linguistic_terms.get(previous_key, current_value)
    return current_value + alpha * (current_value - previous_value)


def transform_to_numerical_matrix(
    raw_matrix: Sequence[Sequence[LinguisticTuple]],
    linguistic_terms: dict[str, int],
) -> DecisionMatrix:
    """Convert the raw 2-tuple linguistic decision matrix into numerical values."""
    return [
        [calculate_numerical_scale(linguistic_terms, term, alpha) for term, alpha in row]
        for row in raw_matrix
    ]


def calculate_todim_dominance_matrix(
    raw_matrix: Sequence[Sequence[LinguisticTuple]],
    linguistic_terms: dict[str, int],
    weights: Sequence[float],
    loss_attenuation_factor: float,
) -> DecisionMatrix:
    """Calculate the overall TODIM relative dominance matrix."""
    weight_sum = sum(weights)
    criterion_count = len(raw_matrix[0])
    learner_count = len(raw_matrix)
    overall_dominance_by_criterion: DecisionMatrix = []

    for criterion_index in range(criterion_count):
        criterion_scores: list[float] = []

        for learner_i in range(learner_count):
            dominance_sum = 0.0
            value_i = calculate_numerical_scale(
                linguistic_terms,
                raw_matrix[learner_i][criterion_index][0],
                raw_matrix[learner_i][criterion_index][1],
            )

            for learner_k in range(learner_count):
                value_k = calculate_numerical_scale(
                    linguistic_terms,
                    raw_matrix[learner_k][criterion_index][0],
                    raw_matrix[learner_k][criterion_index][1],
                )

                if value_i > value_k:
                    dominance = ((weights[criterion_index] / weight_sum) * (value_i - value_k)) ** 0.5
                elif value_i == value_k:
                    dominance = 0.0
                else:
                    dominance = -(
                        (weight_sum / weights[criterion_index]) * (value_k - value_i)
                    ) ** 0.5 / loss_attenuation_factor

                dominance_sum += dominance

            criterion_scores.append(float(round(dominance_sum, 3)))

        overall_dominance_by_criterion.append(criterion_scores)

    return np.asarray(overall_dominance_by_criterion, dtype=float).T.tolist()


def ratio_system(
    decision_matrix: Sequence[Sequence[float]],
    weights: Sequence[float],
    benefit_criteria_count: int,
) -> tuple[list[float], RankedResults, RankedResults]:
    """Calculate learner scores and rankings using the ratio system method."""
    results: RankedResults = []
    values: list[float] = []

    for learner_index, row in enumerate(decision_matrix):
        weighted_values = [value * weight for value, weight in zip(row, weights)]
        score = sum(weighted_values[:benefit_criteria_count]) - sum(
            weighted_values[benefit_criteria_count:]
        )
        rounded_score = float(round(score, 3))
        values.append(rounded_score)
        results.append((learner_name(learner_index), rounded_score))

    sorted_results = sorted(results, key=lambda item: item[1], reverse=True)
    return values, results, sorted_results


def reference_point_method(
    decision_matrix: Sequence[Sequence[float]],
    weights: Sequence[float],
    benefit_criteria_count: int,
) -> tuple[list[float], RankedResults, RankedResults]:
    """Calculate learner scores and rankings using the reference point method."""
    matrix = np.asarray(decision_matrix, dtype=float)
    reference_points: list[float] = []

    for criterion_index, criterion_values in enumerate(matrix.T):
        if criterion_index < benefit_criteria_count:
            reference_points.append(float(np.max(criterion_values)))
        else:
            reference_points.append(float(np.min(criterion_values)))

    values: list[float] = []
    results: RankedResults = []

    for learner_index, row in enumerate(matrix):
        distances = [
            abs(weight * reference_point - weight * value)
            for value, reference_point, weight in zip(row, reference_points, weights)
        ]
        score = float(round(max(distances), 3))
        values.append(score)
        results.append((learner_name(learner_index), score))

    sorted_results = sorted(results, key=lambda item: item[1])
    return values, results, sorted_results


def full_multiplicative_method(
    decision_matrix: Sequence[Sequence[float]],
    weights: Sequence[float],
    benefit_criteria_count: int,
) -> tuple[list[float], RankedResults, RankedResults]:
    """Calculate learner scores and rankings using the full multiplicative method."""
    matrix = np.asarray(decision_matrix, dtype=float)
    min_values = np.min(matrix, axis=0)

    values: list[float] = []
    results: RankedResults = []

    for learner_index, row in enumerate(matrix):
        weighted_values = [
            (value - min_value + 1) ** weight
            for value, min_value, weight in zip(row, min_values, weights)
        ]
        benefit_product = multiply(weighted_values[:benefit_criteria_count])
        cost_product = multiply(weighted_values[benefit_criteria_count:])
        score = float(round(benefit_product / cost_product, 3))
        values.append(score)
        results.append((learner_name(learner_index), score))

    sorted_results = sorted(results, key=lambda item: item[1], reverse=True)
    return values, results, sorted_results


def multiply(values: Iterable[float]) -> float:
    """Return the product of the given values."""
    result = 1.0
    for value in values:
        result *= value
    return result


def rankings_from_sorted_results(sorted_results: RankedResults, learner_count: int) -> list[int]:
    """Convert sorted ranking results into rankings by original learner order."""
    rankings: list[int | None] = [None] * learner_count

    for rank, (learner, _score) in enumerate(sorted_results, start=1):
        learner_index = int(learner.replace("Stud", "")) - 1
        rankings[learner_index] = rank

    if any(rank is None for rank in rankings):
        raise ValueError("Some learners are missing in sorted results.")

    return [int(rank) for rank in rankings]


def standardize_values(values: Sequence[float]) -> list[float]:
    """Standardize one-dimensional evaluation values by Euclidean norm."""
    norm = sum(value**2 for value in values) ** 0.5
    if norm == 0:
        raise ValueError("Cannot standardize values with zero Euclidean norm.")
    return [value / norm for value in values]


def borda_rule_calculation(
    evaluation_values_list: Sequence[Sequence[float]],
    rankings_list: Sequence[Sequence[int]],
) -> tuple[list[float], RankedResults, RankedResults]:
    """Calculate comprehensive evaluation values based on the Borda rule."""
    standardized_ratio = standardize_values(evaluation_values_list[0])
    standardized_reference = standardize_values(evaluation_values_list[1])
    standardized_multiplicative = standardize_values(evaluation_values_list[2])

    learner_count = len(standardized_ratio)
    denominator = learner_count * (learner_count + 1) / 2

    values: list[float] = []
    results: RankedResults = []

    for learner_index in range(learner_count):
        score = (
            standardized_ratio[learner_index]
            * (learner_count - rankings_list[0][learner_index] + 1)
            / denominator
            - standardized_reference[learner_index]
            * rankings_list[1][learner_index]
            / denominator
            + standardized_multiplicative[learner_index]
            * (learner_count - rankings_list[2][learner_index] + 1)
            / denominator
        )
        rounded_score = float(round(score, 3))
        values.append(rounded_score)
        results.append((learner_name(learner_index), rounded_score))

    sorted_results = sorted(results, key=lambda item: item[1], reverse=True)
    return values, results, sorted_results


def classification_rule_01(num_levels: int, max_evaluation_value: float) -> list[float]:
    """Calculate classification boundaries based on the number of levels."""
    return [
        round(((num_levels - level) / num_levels) * max_evaluation_value, 3)
        for level in range(1, num_levels)
    ]


def classify_students(
    evaluation_scores: Sequence[float],
    classification_boundaries: Sequence[float],
) -> dict[str, RankedResults]:
    """Classify students based on evaluation values and classification boundaries."""
    classified_students: dict[str, RankedResults] = {
        f"Level{level}": [] for level in range(1, len(classification_boundaries) + 1)
    }

    for learner_index, score in enumerate(evaluation_scores):
        for level, threshold in enumerate(classification_boundaries, start=1):
            if score <= threshold:
                classified_students[f"Level{level}"].append((learner_name(learner_index), score))
                break

    return classified_students


def classify_learners(
    num_levels: int,
    evaluation_results: Sequence[float],
    logger: ResultLogger,
) -> dict[str, RankedResults]:
    """Classify learners and log the classification results."""
    boundaries = sorted(classification_rule_01(num_levels, max(evaluation_results)))
    boundaries.append(max(evaluation_results))

    classification_result = classify_students(evaluation_results, boundaries)

    logger.write(f"Classification Boundaries (Rule 1): {boundaries}", "Learner Classification Results")
    logger.write("Classification Results:")

    for level, students in classification_result.items():
        if students:
            logger.write(f"{level}: {students}")

    return classification_result


def log_ranked_results(title: str, sorted_results: RankedResults, logger: ResultLogger, rank_label: str = "Rank") -> None:
    """Log sorted ranking results with a consistent format."""
    logger.write(title, title)
    for rank, (learner, score) in enumerate(sorted_results, start=1):
        logger.write(f"{rank_label} {rank}: {learner} = {score}")


def initialize_report(logger: ResultLogger) -> None:
    """Write report metadata to the result file."""
    logger.write("TODIM-MULTMOOA Learner Evaluation Results")
    logger.write(f"Generated Time: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.write("=" * 60)


def run_evaluation(logger: ResultLogger) -> None:
    """Run the full TODIM-MULTMOOA evaluation workflow."""
    initialize_report(logger)

    logger.write("Original Decision Matrix Data (20 learners × 6 evaluation criteria):", "Initial Data")
    for index, row in enumerate(RAW_DECISION_MATRIX):
        logger.write(f"{learner_name(index)}: {row}")

    logger.write(f"Linguistic Term Set Mapping: {LINGUISTIC_TERM_SET}", "Parameter Settings")
    logger.write(f"Weight Vector: {CRITERIA_WEIGHTS}")
    logger.write(f"Theta Parameter: {LOSS_ATTENUATION_FACTOR}")

    numerical_matrix = transform_to_numerical_matrix(
        RAW_DECISION_MATRIX,
        LINGUISTIC_TERM_SET,
    )
    logger.write("Numerical Matrix after NS Transformation:", "NS Value Calculation Results")
    for index, row in enumerate(numerical_matrix):
        logger.write(f"{learner_name(index)}: {[round(value, 4) for value in row]}")

    decision_matrix = calculate_todim_dominance_matrix(
        RAW_DECISION_MATRIX,
        LINGUISTIC_TERM_SET,
        CRITERIA_WEIGHTS,
        LOSS_ATTENUATION_FACTOR,
    )
    logger.write(
        f"TODIM Relative Dominance Matrix calculation completed, "
        f"{len(COMPREHENSIVE_WEIGHTS)} evaluation criteria in total",
        "TODIM Relative Dominance Calculation",
    )

    logger.write("Overall Relative Dominance Matrix:", "Overall Relative Dominance")
    logger.write(np.asarray(decision_matrix))
    logger.write(f"Comprehensive Weight Vector: {COMPREHENSIVE_WEIGHTS}")

    ratio_values, _ratio_results, sorted_ratio_results = ratio_system(
        decision_matrix,
        COMPREHENSIVE_WEIGHTS,
        BENEFIT_CRITERIA_COUNT,
    )
    log_ranked_results("Decision Results and Ranking under Ratio System Method:", sorted_ratio_results, logger)
    logger.write(f"Ratio System Evaluation Values: {ratio_values}")

    reference_values, _reference_results, sorted_reference_results = reference_point_method(
        decision_matrix,
        COMPREHENSIVE_WEIGHTS,
        BENEFIT_CRITERIA_COUNT,
    )
    log_ranked_results(
        "Decision Results and Ranking under Reference Point Method:",
        sorted_reference_results,
        logger,
    )
    logger.write(f"Reference Point Method Evaluation Values: {reference_values}")

    multiplicative_values, _multiplicative_results, sorted_multiplicative_results = (
        full_multiplicative_method(
            decision_matrix,
            COMPREHENSIVE_WEIGHTS,
            BENEFIT_CRITERIA_COUNT,
        )
    )
    log_ranked_results(
        "Decision Results and Ranking under Full Multiplicative Method:",
        sorted_multiplicative_results,
        logger,
    )
    logger.write(f"Full Multiplicative Method Evaluation Values: {multiplicative_values}")

    learner_count = len(ratio_values)
    ratio_rankings = rankings_from_sorted_results(sorted_ratio_results, learner_count)
    reference_rankings = rankings_from_sorted_results(sorted_reference_results, learner_count)
    multiplicative_rankings = rankings_from_sorted_results(sorted_multiplicative_results, learner_count)

    evaluation_values_list = [ratio_values, reference_values, multiplicative_values]
    rankings_list = [ratio_rankings, reference_rankings, multiplicative_rankings]

    logger.write("Ranking Data Imported from Previous Results:", "Borda Rule Input Data")
    logger.write(f"Ratio System Evaluation Values: {ratio_values}")
    logger.write(f"Ratio System Rankings: {ratio_rankings}")
    logger.write(f"Reference Point Method Evaluation Values: {reference_values}")
    logger.write(f"Reference Point Method Rankings: {reference_rankings}")
    logger.write(f"Full Multiplicative Method Evaluation Values: {multiplicative_values}")
    logger.write(f"Full Multiplicative Method Rankings: {multiplicative_rankings}")

    borda_values, _borda_results, sorted_borda_results = borda_rule_calculation(
        evaluation_values_list,
        rankings_list,
    )
    log_ranked_results(
        "Borda Comprehensive Evaluation Results:",
        sorted_borda_results,
        logger,
        rank_label="Overall Rank",
    )
    logger.write(f"Borda Comprehensive Evaluation Values: {borda_values}")

    final_rankings = rankings_from_sorted_results(sorted_borda_results, len(borda_values))
    logger.write("Final Comprehensive Results Imported from Borda:", "Final Comprehensive Results")
    logger.write(f"Final Comprehensive Evaluation Values: {borda_values}")
    logger.write(f"Final Rankings: {final_rankings}")

    final_classification_result = classify_learners(7, borda_values, logger)

    logger.write("\n" + "=" * 60, "Evaluation Summary Report")
    logger.write("Top 5 in Final Comprehensive Ranking:")
    for rank, (learner, score) in enumerate(sorted_borda_results[:5], start=1):
        logger.write(f"Rank {rank}: {learner} (Score: {score})")

    logger.write("\nBottom 5 in Final Comprehensive Ranking:")
    bottom_results = sorted_borda_results[-5:]
    first_bottom_rank = len(sorted_borda_results) - len(bottom_results) + 1
    for rank_offset, (learner, score) in enumerate(bottom_results):
        logger.write(f"Rank {first_bottom_rank + rank_offset}: {learner} (Score: {score})")

    logger.write("\nNumber of Learners by Level:")
    for level, learners in final_classification_result.items():
        if learners:
            logger.write(f"{level}: {len(learners)} learners")

    logger.write(f"\nResult file saved to: {logger.file_path}")
    print(f"\nAll calculation results have been saved to file: {logger.file_path}")


def main() -> None:
    """Script entry point."""
    logger = ResultLogger()
    run_evaluation(logger)


if __name__ == "__main__":
    main()
