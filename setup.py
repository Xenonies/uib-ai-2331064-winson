import sklearn
import matplotlib.pyplot as plt
from sklearn import datasets

# Load iris dataset
iris = datasets.load_iris()
print("Iris data shape:", iris.data.shape)

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target, cmap='viridis')
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("Iris Dataset Scatter Plot")
plt.colorbar()
plt.show()