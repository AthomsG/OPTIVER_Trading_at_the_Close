import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
# For GLM models
import statsmodels.api as sm
from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import probplot
from sklearn.metrics import mean_absolute_error

def display_anova_table(model):
    """
    Display the ANOVA table for the given model.

    Parameters:
    - model: statsmodels regression model
        The fitted regression model.

    Returns:
    None
    """
    anova_table = sm.stats.anova_lm(model)

    # Extract relevant information
    source_info = anova_table.iloc[:-1, :]
    total_info = anova_table.iloc[-1, :]

    # Display the ANOVA table in a formatted way
    print("Analysis of Variance Table\n")
    headers = ["Source", "Df", "SS", "MS", "F", "PR(>F)"]
    table_data = []

    for index, row in source_info.iterrows():
        table_data.append([index, int(row['df']), int(row['sum_sq']), int(row['sum_sq'] / row['df']),
                           round(row['F'], 1), format(row['PR(>F)'], ".2e")])

    table_data.append(["Error", int(total_info['df']), int(total_info['sum_sq']),
                       int(total_info['sum_sq'] / total_info['df']), "", ""])
    
    table_data.append(["Total", int(total_info['df'] + source_info['df'].sum()),
                       int(total_info['sum_sq'] + source_info['sum_sq'].sum()),
                       int((total_info['sum_sq'] + source_info['sum_sq'].sum()) / 
                           (total_info['df'] + source_info['df'].sum())), "", ""])

    print(tabulate(table_data, headers=headers, tablefmt="pretty"))


def residuals_analysis(residuals):
    """
    Plot the histogram, boxplot, QQ plot, and perform Lilliefors test on residuals.

    Parameters:
    - residuals: array-like
        The residuals from a linear model.

    Returns:
    None
    """
    # Normalize the residuals
    normalized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)

    # Create a custom grid for subplots
    plt.figure(figsize=(10, 8))
    grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.3)

    # Plot histogram on the top left subplot
    plt.subplot(grid[0, 0])
    plt.hist(normalized_residuals, bins='auto', density=True, alpha=0.7)
    plt.title('Histogram of Normalized Residuals')
    plt.xlabel('Normalized Residuals')
    plt.ylabel('Density')

    # Set limits for x-axis in the histogram
    plt.xlim(-4, 4)

    # Plot vertical boxplot on the top right subplot
    plt.subplot(grid[0, 1])
    plt.boxplot(normalized_residuals, vert=True)
    plt.title('Boxplot of Normalized Residuals')
    plt.xlabel('Normalized Residuals')

    # QQ plot on the bottom, spanning two columns
    plt.subplot(grid[1, :])
    probplot(normalized_residuals, plot=plt)
    plt.title('QQ Plot of Normalized Residuals')

    # Add a line with slope 45 degrees
    plt.plot([-4, 4], [-4, 4], color='black', linestyle='--')

    # Perform Lilliefors test
    lilliefors_statistic, lilliefors_p_value = lilliefors(residuals)

    # Print the test results
    print(f"Lilliefors Statistic: {lilliefors_statistic}")
    print(f"P-value: {lilliefors_p_value}")

    plt.show()


def find_highest_p_value(model):
    """
    Find the variable with the highest p-value in the ANOVA table.

    Parameters:
    - model: statsmodels regression model
        The fitted regression model.

    Returns:
    None
    """
    anova_table = sm.stats.anova_lm(model)

    # Exclude the last row (Total)
    source_info = anova_table.iloc[:-1, :]

    # Find the variable with the highest p-value
    max_p_value_row = source_info.loc[source_info['PR(>F)'].idxmax()]

    # Extract variable name and corresponding p-value
    variable_name = max_p_value_row.name
    p_value = max_p_value_row['PR(>F)']

    # Print the results
    print(f"Variable with the highest P value: {variable_name}")
    print(f"P value: {p_value:.4e}")


def prediction_residuals_and_errors(model, X_test, y_test):
    """
    Plot residuals, boxplot of normalized residuals, and print error metrics.

    Parameters:
    - model: sklearn or statsmodels regression model
        The fitted regression model.

    Returns:
    None
    """

    # Compute predictions
    y_pred = model.predict(X_test)

    # Compute residuals
    residuals = y_pred - y_test

    # Normalize residuals
    normalized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)

    # Custom grid for subplots
    plt.figure(figsize=(12, 6))
    grid = plt.GridSpec(1, 2, wspace=0.4)

    # Scatter plot on the left subplot
    plt.subplot(grid[0])
    plt.scatter(y_pred, y_test, alpha=0.4)
    plt.xlabel('Predicted Target')
    plt.ylabel('Target value')
    plt.grid(color='gray', alpha=0.5)
    plt.plot([-40, 40], [-40, 40], color='red', ls='--')
    plt.title('Scatter Plot')

    # Boxplot of normalized residuals on the right subplot
    plt.subplot(grid[1])
    plt.boxplot(normalized_residuals, vert=True)
    plt.title('Boxplot of Normalized Residuals')
    plt.ylabel('Normalized Residuals (Squared)')

    plt.show()

    # Mean Absolute Error
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae:.4f}")

    # Squared Error
    squared_error = np.mean((y_pred - y_test)**2)
    print(f"Mean Squared Error:  {squared_error:.4f}")