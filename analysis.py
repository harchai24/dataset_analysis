import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
data = iris.data

# Create a Pandas DataFrame for easier analysis
iris_df = pd.DataFrame(data, columns=iris.feature_names)

# Calculate the mean, median, and standard deviation for each feature
means = iris_df.mean()
medians = iris_df.median()
std_devs = iris_df.std()

# Create histograms to visualize data distribution
iris_df.hist(bins=20, figsize=(12, 8))
plt.suptitle('Data Distribution')
plt.show()

# Create a bar chart to show means for different variables
means.plot(kind='bar', figsize=(10, 6))
plt.title('Mean Values for Numeric Variables')
plt.ylabel('Mean')
plt.show()

# Display summary statistics
print("Summary Statistics:")
print("Means:\n", means)
print("\nMedians:\n", medians)
print("\nStandard Deviation:\n", std_devs)

# Save the Pandas DataFrame to a CSV file (Optional)
# iris_df.to_csv('iris_dataset_summary.csv', index=False)

# Documentation:
# Introduction
print("\nIntroduction:")
print("The Iris dataset is a well-known dataset used for basic data analysis, containing measurements of iris flowers.")

# Descriptive Analysis
print("\nDescriptive Analysis:")
print("The dataset's numeric features exhibit varying central tendencies and spreads.")
print("Sepal Length and Sepal Width have moderate variability, while Petal Length and Petal Width show more substantial variability.")

# References
print("\nReferences:")
print("- Scikit-Learn - Iris Dataset: https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html")
print("- Pandas Documentation: https://pandas.pydata.org/")
print("- Matplotlib Documentation: https://matplotlib.org/")
