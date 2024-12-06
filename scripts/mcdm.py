import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def calculate_metrics(df, true_labels, model_names):
    # Initialize list to collect metrics
    metrics_data = []

    # Iterate over each class
    for class_label in true_labels.unique():
        # Iterate over each model
        for model_name in model_names:
            # Convert probabilities to predicted binary labels (the threshold = 0.5)
            predicted_labels = df[model_name] >= 0.5

            # Calculate true positive (TP), false positive (FP), true negative (TN), false negative (FN)
            TP = ((predicted_labels == 1) & (true_labels == class_label)).sum()
            FP = ((predicted_labels == 1) & (true_labels != class_label)).sum()
            TN = ((predicted_labels == 0) & (true_labels != class_label)).sum()
            FN = ((predicted_labels == 0) & (true_labels == class_label)).sum()

            # Calculate metrics
            accuracy = (TP + TN) / (TP + FP + TN + FN) if TP + FP + TN + FN != 0 else 0
            precision = TP / (TP + FP) if TP + FP != 0 else 0
            recall = TP / (TP + FN) if TP + FN != 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
            f2_score = 5 * precision * recall / (4 * precision + recall) if precision + recall != 0 else 0

            # Collect metrics in a dictionary
            metrics_dict = {
                'Model': model_name,
                'Class': class_label,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_score': f1_score,
                'F2_score': f2_score
            }

            # Append metrics dictionary to list
            metrics_data.append(metrics_dict)

    # Create DataFrame from list of metrics dictionaries
    metrics_df = pd.DataFrame(metrics_data)

    return metrics_df

def calculate_decision_maker_weights(index_matrix):
    """
    Calculate the weights of decision-makers based on the index matrix using the Entropy Weight Method.

    Parameters:
        index_matrix (numpy.ndarray): The index matrix with dimensions (num_alternatives, num_decision_makers).

    Returns:
        dm_weights (numpy.ndarray): Array containing the weights of decision-makers.
    """
    # Normalize the index matrix
    normalized_index_matrix = index_matrix.transpose()   / np.sum(index_matrix, axis=1)

    # Apply emphasis on recall ratio
    normalized_index_matrix[1, :] *= 100  # Assuming recall is at index 1, multiply its values by 2 to emphasize its importance

    # Calculate the weighted normalized index matrix
    weighted_normalized_index_matrix = normalized_index_matrix

    # Calculate the information entropy for each decision-maker
    entropy = -np.sum(weighted_normalized_index_matrix * np.log2(weighted_normalized_index_matrix), axis=0)

    # Calculate the weight of each decision-maker
    dm_weights = (1 - entropy) / np.sum(1 - entropy)

    return dm_weights

def topsis(data_matrix, weights):
    # Step 1: Normalize the data matrix
    normalized_matrix = data_matrix / np.linalg.norm(data_matrix, axis=0)

    # Step 2: Calculate the ideal solution and anti-ideal solution
    ideal_solution = np.max(normalized_matrix, axis=0)
    anti_ideal_solution = np.min(normalized_matrix, axis=0)

    # Step 3: Calculate the distance from each alternative to the ideal and anti-ideal solutions
    distance_to_ideal = np.linalg.norm(normalized_matrix - ideal_solution, axis=1)
    distance_to_anti_ideal = np.linalg.norm(normalized_matrix - anti_ideal_solution, axis=1)

    # Step 4: Calculate the TOPSIS score (closeness to the ideal solution)
    topsis_score = distance_to_anti_ideal / (distance_to_ideal + distance_to_anti_ideal)
    normalized_topsis_score = topsis_score / np.sum(topsis_score)
    # Rank alternatives based on TOPSIS score (higher score = better)
    ranked_indices = np.argsort(normalized_topsis_score)[::-1]  # Sort in descending order
    topsis_df = pd.DataFrame({
            'TOPSIS Score': normalized_topsis_score
        })

    # Save TOPSIS scores to Excel file
    topsis_df.to_excel('topsis_scores.xlsx', index=False)
    return ranked_indices,normalized_topsis_score

def test_average_significance_w(data, target_mean, alpha):
    # Calculate the sample size
    n = len(data)

    # Calculate the mean of the sample

    sample_mean = np.mean(data)
    data = np.array(data)
    # Perform one-sample Wilcoxon signed-rank test
    # _, p_value = stats.wilcoxon(data - target_mean, alternative='less')
    _, p_value = stats.wilcoxon(data - target_mean, alternative='less')



    # Create a KDE plot
    plt.hist(data, bins=10, density=True, alpha=0.5, color='b', label='Data')
    plt.axvline(sample_mean, color='r', linestyle='dashed', linewidth=1, label='Predicted mean')
    kde = stats.gaussian_kde(data)
    kde_xs = np.linspace(0, 1, 1000)
    plt.plot(kde_xs, kde.pdf(kde_xs), label='Fitted Distribution', color='g')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Histogram and Fitted Distribution of Data')
    plt.legend()

    # Annotate p-value
    plt.annotate(f'p-value = {p_value:.4f}', xy=(0.5, 0.5), xytext=(0.6, 0.6),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 )

    # Show plot
    plt.show()

    # Print results
    print("Predicted average:", sample_mean)
    print("Null Hypothesis (Threshold to determine if the CO2 is leaking):", target_mean)
    print("p-value:", p_value)

    # Check if the p-value is less than a significance level (e.g., 0.05)
    print("#" * 100)
    if p_value < alpha:

        print("The average is significantly less than the threshold with probability {:4f}%, the prediction result is confident, no leaking risk".format((1-alpha)*100))
    else:
        print("The average is not significantly less than the threshold, we cannot determine if the CO2 is leaking")
    print("#" * 100)


def test_average_significance_t(data, target_mean, alpha):
    # Calculate the sample size
    n = len(data)
  
    # Calculate the mean of the sample
       
    data = np.where(data >= 1, 1, data)

    # Calculate the mean
    sample_mean = np.mean(data)
 
    # Perform one-sample t-test for one-sided alternative (less)
    t_statistic, p_value = stats.ttest_1samp(data, target_mean, alternative='less')

  # Create a KDE plot
    plt.hist(data, bins=10, density=True, alpha=0.5, color='b', label='Data')
    plt.axvline(sample_mean, color='r', linestyle='dashed', linewidth=1, label='Predicted mean')
    kde = stats.gaussian_kde(data)
    kde_xs = np.linspace(np.min(data), 1, 1000)
    plt.plot(kde_xs, kde.pdf(kde_xs), label='Fitted Distribution', color='g')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Histogram and Fitted Distribution of Data')
    plt.legend()
   # Annotate p-value
    plt.annotate(f'p-value = {p_value:.3f}', xy=(0.5, 0.5), xytext=(0.6, 0.6),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 )

    # Show plot
    plt.show()


    # Print results
    print("predict average:", sample_mean)
    print("Null Hypothesis (Threshod to determine if the co2 is leaking):", target_mean)
    print("t-statistic:", t_statistic)
    print("p-value:", p_value)

    # Check if the p-value is less than a significance level (e.g., 0.05)
    print("#" * 100)
    if p_value < alpha:
       print("The average is significantly less than {} with probability {:4f}%, the predict result is confident, no leaking risk".format(target_mean,(1-alpha)*100))
    else:
       print("The average is not significantly less than {}, we cannot determine if the CO2 is leaking".format(target_mean))
    print("#" * 100)


def get_model_performance(data,weights=None,threshold = 0.5):
    # Calculate True Positive (TP), False Positive (FP), True Negative (TN), False Negative (FN)
    models_columns = data.iloc[:, 1:4]

    # Multiply each column of the previous 12 columns with corresponding weights


    # Average the values across each row
    if weights is not None:
      weighted_data = models_columns * weights
      average = weighted_data.sum(axis=1)

    else:
      weighted_data = models_columns
      average = weighted_data.mean(axis=1)


    # Apply the condition: if average is greater than 0.5, set to 1, else set to 0
    pred = np.where(average >= threshold, 1, 0)

    # Add the averaged row values as a new column to the DataFrame
    data['Predicted'] = pred

    # Calculate True Positive (TP), False Positive (FP), True Negative (TN), False Negative (FN)
    TP = ((data['Predicted'] ==1 ) & (data['label'] == 1)).sum()
    FP = ((data['Predicted'] == 1) & (data['label'] == 0)).sum()
    FP_indices = data.loc[(data['Predicted'] == 1) & (data['label'] == 0)].index

    TN = ((data['Predicted'] == 0) & (data['label'] == 0)).sum()
    FN = ((data['Predicted'] ==0) & (data['label'] == 1)).sum()

    # Calculate precision, recall, and accuracy

    accuracy = (TP + TN) / (TP + FP + TN + FN) if TP + FP + TN + FN != 0 else 0
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    f2_score = 5 * precision * recall / (4 * precision + recall) if precision + recall != 0 else 0

    return pred, accuracy, precision,recall,  f1_score, f2_score

def MC_dropData(model_paths, topsis_score,index):
    # Load data from Excel file
    data = pd.read_excel(model_paths)
    probabilities = data.iloc[index,1:31]

    # 3 is total model numbers
    weighted_probability = probabilities*topsis_score*3
  
    # print(weights-probabilities)
    return weighted_probability


def probability_samples(model_paths, topsis_scores, index):
    samples =[]
    for i in range(len(model_paths)):
      topsis_score = topsis_scores[i]
      samples.append(MC_dropData(model_paths[i], topsis_score,index))

    samples_list = [item for sublist in samples for item in sublist]
    return samples_list
