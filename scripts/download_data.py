from sklearn.datasets import load_breast_cancer
import pandas as pd


iris = load_breast_cancer()

data = pd.DataFrame(iris["data"], columns=iris["feature_names"])
target = pd.DataFrame(iris["target"], columns=["label"])

data.to_csv('../data/data.csv', index=False)
target.to_csv('../data/labels.csv', index=False)

print("Data downloaded")