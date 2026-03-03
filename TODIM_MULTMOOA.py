"""
TODIM-MULTMOOA Evaluation System for All Learners
Author: CTBU_XuBJ
Date: January 6, 2025
Modified: Added file output functionality
"""
import numpy as np
import datetime
import os

# Create output directory for results
output_dir = "evaluation_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create result file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
result_file = os.path.join(output_dir, f"TODIM_MULTMOOA_Results_{timestamp}.txt")

def write_to_file(content, file_path=result_file):
    """Write content to file"""
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(str(content) + '\n')

def print_and_save(content, title=""):
    """Print to console and save to file simultaneously"""
    if title:
        print(f"\n=== {title} ===")
        write_to_file(f"\n=== {title} ===")
    print(content)
    write_to_file(content)

# Initialize result file
write_to_file(f"TODIM-MULTMOOA Learner Evaluation Results")
write_to_file(f"Generated Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
write_to_file("="*60)

# Decision matrix data (20 learners × 6 evaluation criteria)
dm = [[('s_3', -0.28), ('s_0', 0.23), ('s_2', 0.27), ('s_5', 0.44), ('s_3', 0.45), ('s_5', 0.28)],
      [('s_2', 0.31), ('s_0', 0.43), ('s_4', 0.37), ('s_5', -0.43), ('s_6', 0), ('s_5', 0.39)],
      [('s_5', 0.25), ('s_4', 0.42), ('s_6', 0), ('s_5', 0.46), ('s_0', 0.44), ('s_4', 0.29)],
      [('s_5', 0.21), ('s_1', 0.22), ('s_3', 0.18), ('s_0', 0.29), ('s_1', 0.13), ('s_2', 0.19)],
      [('s_1', 0.48), ('s_5', -0.16), ('s_5', 0.34), ('s_0', 0.22), ('s_2', 0.27), ('s_4', 0.19)],
      [('s_4', 0.15), ('s_0', 0.38), ('s_2', 0.27), ('s_4', -0.24), ('s_1', 0.18), ('s_0', 0.43)],
      [('s_1', 0.31), ('s_5', 0.37), ('s_0', 0.4), ('s_6', 0), ('s_2', 0.32), ('s_1', 0.45)],
      [('s_6', 0), ('s_4', -0.25), ('s_4', 0.46), ('s_4', 0.44), ('s_6', 0), ('s_3', 0.25)],
      [('s_6', 0.11), ('s_1', 0.21), ('s_5', 0.2), ('s_5', 0.24), ('s_0', 0.43), ('s_0', 0.41)],
      [('s_2', 0.44), ('s_0', 0.26), ('s_6', 0), ('s_1', 0.39), ('s_3', 0.31), ('s_1', 0.28)],
      [('s_0', 0.44), ('s_2', 0.12), ('s_3', 0.4), ('s_2', 0.47), ('s_3', 0.49), ('s_6', 0)],
      [('s_4', 0.38), ('s_1', 0.21), ('s_5', -0.28), ('s_4', 0.43), ('s_3', 0.19), ('s_3', 0.46)],
      [('s_6', 0), ('s_1', 0.22), ('s_2', 0.2), ('s_5', 0.4), ('s_6', 0), ('s_0', 0.16)],
      [('s_4', 0.44), ('s_1', 0.34), ('s_1', 0.28), ('s_0', 0.23), ('s_5', 0.3), ('s_1', 0.2)],
      [('s_3', 0.29), ('s_0', 0.43), ('s_2', 0.23), ('s_6', 0), ('s_5', 0.47), ('s_5', 0.46)],
      [('s_6', 0), ('s_2', 0.26), ('s_1', 0.22), ('s_3', 0.3), ('s_4', 0.5), ('s_1', 0.21)],
      [('s_5', 0.39), ('s_4', 0.46), ('s_2', 0.46), ('s_0', 0.33), ('s_3', 0.17), ('s_6', 0)],
      [('s_5', 0.19), ('s_1', 0.24), ('s_5', 0.3), ('s_1', 0.25), ('s_1', 0.14), ('s_2', 0.4)],
      [('s_5', 0.23), ('s_5', 0.31), ('s_6', 0), ('s_3', 0.44), ('s_0', 0.31), ('s_2', 0.42)],
      [('s_0', 0.21), ('s_1', 0.38), ('s_0', 0.3), ('s_0', 0.39), ('s_1', 0.12), ('s_1', 0.3)]]

print_and_save(f"Original Decision Matrix Data (20 learners × 6 evaluation criteria):", "Initial Data")
for i, row in enumerate(dm):
    print_and_save(f"Stud{i+1}: {row}")

# Linguistic term set mapping
S = {'s_0': 0, 's_1': 1, 's_2': 2, 's_3': 3, 's_4': 4, 's_5': 5, 's_6': 6}
# Weight vector for criteria
wjr = [1, 0.2233, 0.4433, 0.6667, 0.3333, 0.6667]
# Theta parameter
theta = 2

print_and_save(f"Linguistic Term Set Mapping: {S}", "Parameter Settings")
print_and_save(f"Weight Vector: {wjr}")
print_and_save(f"Theta Parameter: {theta}")

def NS(S, s, theta):
    '''
    Definition 2.3 ([44]). Let S be a linguistic term set. The 2-tuple numerical scale
    NS: S × [−0.5, 0.5) → R is defined as:
    :param S: Defined binary linguistic term set
    :param s: Linguistic evaluation value
    :param theta: Binary linguistic evaluation value
    :return: NS value
    '''
    # Calculation formula
    if theta >= 0 and f"s_{int(s.split('_')[1]) + 1}" in S:
        ns = S[s] + theta * (S[f"s_{int(s.split('_')[1]) + 1}"] - S[s])
    else:
        ns = S[s] + theta * (S[s] - S[f"s_{int(s.split('_')[1]) - 1}"])
    return ns

# Calculate NS values
numerical_matrix = []
for item in dm:
    row_values = []
    for item_ij in item:
        row_values.append(NS(S, item_ij[0], item_ij[1]))
    numerical_matrix.append(row_values)

print_and_save("Numerical Matrix after NS Transformation:", "NS Value Calculation Results")
for i, row in enumerate(numerical_matrix):
    formatted_row = [round(val, 4) for val in row]
    print_and_save(f"Stud{i+1}: {formatted_row}")

# Calculate relative importance in TODIM
dominance_matrices = []
# Calculate relative dominance between any two learners on the j-th evaluation criterion
for j in range(len(dm[0])):  # j-th evaluation criterion
    criterion_dominance = []  # Relative dominance of alternative i over alternative a on j-th criterion
    for item1 in dm:  # Extract i-th evaluation criterion
        relative_dominance = []  # Relative dominance of alternative i over alternative a
        for item2 in dm:  # Extract a-th evaluation criterion
            # Calculation formula
            if NS(S, item1[j][0], item1[j][1]) > NS(S, item2[j][0], item2[j][1]):
                res = ((wjr[j] / sum(wjr)) * (NS(S, item1[j][0], item1[j][1]) - NS(S, item2[j][0], item2[j][1]))) ** (
                    0.5)
            elif NS(S, item1[j][0], item1[j][1]) == NS(S, item2[j][0], item2[j][1]):
                res = 0
            else:
                res = -(1 / theta) * ((sum(wjr) / wjr[j]) * (
                        NS(S, item2[j][0], item2[j][1]) - NS(S, item1[j][0], item1[j][1]))) ** (
                          0.5)
            relative_dominance.append(res)
        criterion_dominance.append(relative_dominance)
    dominance_matrices.append(criterion_dominance)

print_and_save(f"TODIM Relative Dominance Matrix calculation completed, {len(dominance_matrices)} evaluation criteria in total", "TODIM Relative Dominance Calculation")

# Calculate overall relative dominance to get an M×N matrix
overall_dominance = []
for criterion_matrix in dominance_matrices:
    criterion_sums = []
    for alternative_dominance in criterion_matrix:
        total_dominance = sum(alternative_dominance)
        criterion_sums.append(round(total_dominance, 3))
    overall_dominance.append(criterion_sums)

decision_matrix = np.mat(overall_dominance).T.tolist()
# Comprehensive weight vector
comprehensive_weights = [0.3, 0.067, 0.133, 0.2, 0.1, 0.2]

print_and_save("Overall Relative Dominance Matrix:", "Overall Relative Dominance")
print_and_save(np.mat(decision_matrix))
print_and_save(f"Comprehensive Weight Vector: {comprehensive_weights}")

# Step 3.2: Calculate Ratio System Results
def ratio_system(decision_matrix, weights, benefit_criteria_count):
    '''
    Calculate decision results and ranking under ratio system
    :param decision_matrix: Standardized decision likelihood matrix
    :param weights: Comprehensive weights
    :param benefit_criteria_count: Starting position of cost-type criteria
    :return: Decision results and ranking values
    '''
    ratio_results = []
    ratio_values = []
    for r in range(len(decision_matrix)):  # Extract r-th alternative from standardized decision matrix
        weighted_values = []
        for value, weight in zip(decision_matrix[r], weights):  # Extract each evaluation value and corresponding weight
            weighted_values.append(value * weight)
        result = sum(weighted_values[:benefit_criteria_count]) - sum(weighted_values[benefit_criteria_count:])
        ratio_results.append(('Stud' + str(r + 1), round(result, 3)))  # L for Learner
        ratio_values.append(round(result, 3))
    sorted_ratio_results = sorted(ratio_results, key=lambda x: x[1], reverse=True)  # Sort evaluation results
    return ratio_values, ratio_results, sorted_ratio_results

ratio_values, ratio_results, sorted_ratio_results = ratio_system(decision_matrix, comprehensive_weights, 7)
print_and_save('Decision Results and Ranking under Ratio System Method:', "Ratio System Method Results")
for i, (learner, score) in enumerate(sorted_ratio_results):
    print_and_save(f"Rank {i+1}: {learner} = {score}")
print_and_save(f"Ratio System Evaluation Values: {ratio_values}")

# Step 3.3: Calculate Reference Point Method Results
def reference_point_method(decision_matrix, weights, benefit_criteria_count):
    '''
    Calculate reference point method results
    :param decision_matrix: Standardized likelihood decision matrix
    :param weights: Comprehensive weights
    :param benefit_criteria_count: Position of cost-type criteria
    :return: Decision values and ranking results under reference point method
    '''
    reference_points = []
    # Calculate reference points
    for i in range(len(np.mat(decision_matrix).T.tolist())):
        if i < benefit_criteria_count:
            reference_points.append(max(np.mat(decision_matrix).T.tolist()[i]))
        else:
            reference_points.append(min(np.mat(decision_matrix).T.tolist()[i]))
    # Calculate distances
    reference_results = []
    reference_values = []
    for i in range(len(decision_matrix)):
        distances = []
        for j in range(len(decision_matrix[i])):
            distances.append(abs(weights[j] * reference_points[j] - weights[j] * decision_matrix[i][j]))
        reference_results.append(('Stud' + str(i + 1), round(max(distances), 3)))
        reference_values.append(round(max(distances), 3))
    sorted_reference_results = sorted(reference_results, key=lambda x: x[1])  # Sort evaluation results
    return reference_values, reference_results, sorted_reference_results

reference_values, reference_results, sorted_reference_results = reference_point_method(decision_matrix, comprehensive_weights, 7)
print_and_save('Decision Results and Ranking under Reference Point Method:', "Reference Point Method Results")
for i, (learner, score) in enumerate(sorted_reference_results):
    print_and_save(f"Rank {i+1}: {learner} = {score}")
print_and_save(f"Reference Point Method Evaluation Values: {reference_values}")

# Step 3.4: Calculate Full Multiplicative Method Results
def full_multiplicative_method(decision_matrix, weights, benefit_criteria_count):
    '''
    Calculate decision results and ranking under full multiplicative method
    :param decision_matrix: Standardized decision likelihood matrix
    :param weights: Comprehensive weights
    :param benefit_criteria_count: Starting position of cost-type criteria
    :return: Decision results and ranking values
    '''
    # Get minimum values for each criterion
    min_values = []
    for criterion_values in np.mat(decision_matrix).T.tolist():
        min_values.append(min(criterion_values))

    # Perform full multiplicative ranking
    multiplicative_results = []
    multiplicative_values = []
    for r in range(len(decision_matrix)):  # Extract r-th alternative from standardized decision matrix
        weighted_values = []
        for value, min_val, weight in zip(decision_matrix[r], min_values, weights):
            weighted_values.append((value - min_val + 1) ** weight)

        # Calculate benefit criteria product
        benefit_product = 1
        for item in weighted_values[:benefit_criteria_count]:
            benefit_product *= item

        # Calculate cost criteria product
        cost_product = 1
        for item in weighted_values[benefit_criteria_count:]:
            cost_product *= item

        result = benefit_product / cost_product
        multiplicative_results.append(('Stud' + str(r + 1), round(result, 3)))
        multiplicative_values.append(round(result, 3))

    sorted_multiplicative_results = sorted(multiplicative_results, key=lambda x: x[1], reverse=True)
    return multiplicative_values, multiplicative_results, sorted_multiplicative_results

multiplicative_values, multiplicative_results, sorted_multiplicative_results = full_multiplicative_method(decision_matrix, comprehensive_weights, 7)
print_and_save('Decision Results and Ranking under Full Multiplicative Method:', "Full Multiplicative Method Results")
for i, (learner, score) in enumerate(sorted_multiplicative_results):
    print_and_save(f"Rank {i+1}: {learner} = {score}")
print_and_save(f"Full Multiplicative Method Evaluation Values: {multiplicative_values}")

# Borda Rule Calculation
# Using provided evaluation values and rankings
ratio_evaluation_values = [-7.692, -5.314, 1.49, -14.529, -14.769, -17.614, -16.205, 3.394, -6.225, -16.673, -13.125, -4.232, -5.029,
       -17.399, -3.636, -8.266, -5.039, -10.01, -4.671, -31.223]
ratio_rankings = [10, 8, 2, 14, 15, 19, 16, 1, 9, 17, 13, 4, 6, 18, 3, 11, 7, 12, 5, 20]

reference_evaluation_values = [8.638, 9.694, 5.4, 9.174, 11.451, 8.949, 11.886, 3.789, 9.021, 9.322, 13.605, 4.894, 9.582, 9.366, 7.407, 7.331,
      9.045, 7.555, 5.646, 14.13]
reference_rankings = [4, 19, 17, 11, 16, 14, 18, 1, 10, 12, 9, 3, 15, 13, 7, 6, 8, 2, 5, 20]

multiplicative_evaluation_values = [24.162, 28.044, 33.535, 16.406, 12.412, 15.918, 14.409, 40.627, 20.36, 16.444, 15.444, 33.367, 18.705, 12.401,
      30.504, 25.344, 23.423, 24.231, 24.621, 3.5]
multiplicative_rankings = [9, 5, 2, 14, 18, 15, 17, 1, 11, 13, 16, 3, 12, 19, 4, 6, 10, 8, 7, 20]

evaluation_values_list = [ratio_evaluation_values, reference_evaluation_values, multiplicative_evaluation_values]
rankings_list = [ratio_rankings, reference_rankings, multiplicative_rankings]

print_and_save("Ranking Data from Three Methods:", "Borda Rule Input Data")
print_and_save(f"Ratio System Rankings: {ratio_rankings}")
print_and_save(f"Reference Point Method Rankings: {reference_rankings}")
print_and_save(f"Full Multiplicative Method Rankings: {multiplicative_rankings}")

def standardize_values(values):
    '''
    Standardize evaluation values from three methods
    :param values: Decision values from each method, one-dimensional list
    :return: Standardized decision values, one-dimensional list
    '''
    sum_of_squares = 0
    for item in values:
        sum_of_squares += item ** 2
    standardized_values = []
    for item in values:
        standardized_values.append(item / (sum_of_squares ** (1 / 2)))
    return standardized_values

def borda_rule_calculation(evaluation_values_list: list, rankings_list: list):
    '''
    Calculate comprehensive evaluation values based on Borda rule
    :param evaluation_values_list: Decision values list
    :param rankings_list: Rankings list
    :return: Borda comprehensive results list
    '''
    standardized_ratio = standardize_values(evaluation_values_list[0])
    standardized_reference = standardize_values(evaluation_values_list[1])
    standardized_multiplicative = standardize_values(evaluation_values_list[2])

    borda_results = []
    borda_values = []

    for learner_index in range(len(standardized_ratio)):
        # Borda rule calculation formula
        result = (standardized_ratio[learner_index] * (len(standardized_ratio) - rankings_list[0][learner_index] + 1) /
                 (len(standardized_ratio) * (len(standardized_ratio) + 1) / 2) -
                 standardized_reference[learner_index] * (rankings_list[1][learner_index]) /
                 (len(standardized_reference) * (len(standardized_reference) + 1) / 2) +
                 standardized_multiplicative[learner_index] * (len(standardized_multiplicative) - rankings_list[2][learner_index] + 1) /
                 (len(standardized_multiplicative) * (len(standardized_multiplicative) + 1) / 2))

        borda_values.append(round(result, 3))
        borda_results.append(('Stud' + str(learner_index + 1), round(result, 3)))

    sorted_borda_results = sorted(borda_results, key=lambda x: x[1], reverse=True)
    return borda_values, borda_results, sorted_borda_results

borda_values, borda_results, sorted_borda_results = borda_rule_calculation(evaluation_values_list, rankings_list)

print_and_save('Borda Comprehensive Evaluation Results:', "Borda Comprehensive Evaluation")
for i, (learner, score) in enumerate(sorted_borda_results):
    print_and_save(f"Overall Rank {i+1}: {learner} = {score}")
print_and_save(f'Borda Comprehensive Evaluation Values: {borda_values}')

# Final comprehensive evaluation values and rankings
comprehensive_evaluation_values = [0.002, -0.007, 0.021, -0.015, -0.027, -0.013, -0.029, 0.042, -0.008, -0.013, -0.02, 0.02, -0.015, -0.017, 0.012,
       0.005, -0.003, 0.005, 0.006, -0.035]
final_rankings = [8, 19, 2, 14, 16, 12, 17, 1, 11, 12, 13, 3, 14, 15, 6, 9, 18, 7, 10, 20]

def classify_students(evaluation_scores, classification_boundaries):
    '''
    Classify students based on their proximity scores and boundary proximity scores
    :param evaluation_scores: Student evaluation values, one-dimensional list
    :param classification_boundaries: Classification thresholds, one-dimensional list
    :return: Sorting and classification results
    '''
    # Preprocess student proximity scores, establish one-to-one correspondence between students and proximity scores
    preprocessed_scores = []
    for i in range(len(evaluation_scores)):
        preprocessed_scores.append(('Stud' + str(i + 1), evaluation_scores[i]))

    # Use classification_boundaries values as classification standards
    boundary_thresholds = classification_boundaries
    # Create categories based on boundary values
    classified_students = {'Level' + str(len(boundary_thresholds) - i): [] for i in range(len(boundary_thresholds))}

    for i in range(len(preprocessed_scores)):
        for j, threshold in enumerate(boundary_thresholds, 1):
            if preprocessed_scores[i][1] <= threshold:
                classified_students['Level' + str(j)].append(preprocessed_scores[i])
                break

    return classified_students

def classification_rule_01(num_levels, max_evaluation_value):
    '''
    Classification Rule 1: Classification based on number of levels
    :param num_levels: Number of levels, integer type
    :param max_evaluation_value: Maximum evaluation value
    :return: Classification boundaries, one-dimensional list
    '''
    classification_boundaries = []
    for i in range(num_levels):
        if i + 1 < num_levels:
            classification_boundaries.append(round(((num_levels - (i + 1)) / num_levels) * max_evaluation_value, 3))
        else:
            continue
    return classification_boundaries

def classify_learners_function(num_levels, evaluation_results):
    '''
    Level-based classification method
    :param num_levels: Number of levels to classify into
    :param evaluation_results: Evaluation values, one-dimensional list
    :return: Classification results
    '''
    classification_boundaries = classification_rule_01(num_levels, max(evaluation_results))
    processed_boundaries = sorted(classification_boundaries, reverse=False)
    processed_boundaries.append(max(evaluation_results))
    classification_result = classify_students(evaluation_results, processed_boundaries)

    print_and_save(f'Classification Boundaries (Rule 1): {processed_boundaries}', "Learner Classification Results")
    print_and_save('Classification Results:')
    for level, students in classification_result.items():
        if students:  # Only display non-empty levels
            print_and_save(f"{level}: {students}")

    return classification_result

# Perform classification
final_classification_result = classify_learners_function(7, comprehensive_evaluation_values)

# Generate summary report
print_and_save("\n" + "="*60, "Evaluation Summary Report")
print_and_save("Top 5 in Final Comprehensive Ranking:")
for i in range(min(5, len(sorted_borda_results))):
    learner, score = sorted_borda_results[i]
    print_and_save(f"Rank {i+1}: {learner} (Score: {score})")

print_and_save("\nBottom 5 in Final Comprehensive Ranking:")
for i in range(max(0, len(sorted_borda_results)-5), len(sorted_borda_results)):
    learner, score = sorted_borda_results[i]
    print_and_save(f"Rank {i+1}: {learner} (Score: {score})")

# Statistics of learners in each classification level
print_and_save("\nNumber of Learners by Level:")
for level in sorted(final_classification_result.keys()):
    count = len(final_classification_result[level])
    if count > 0:
        print_and_save(f"{level}: {count} learners")

print_and_save(f"\nResult file saved to: {result_file}")
print(f"\nAll calculation results have been saved to file: {result_file}")