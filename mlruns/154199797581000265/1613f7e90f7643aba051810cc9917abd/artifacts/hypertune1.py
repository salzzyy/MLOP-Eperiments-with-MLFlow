import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load Breast Cancer Dataset
data = load_breast_cancer()
x = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# Ensure that all integer columns are converted to float64
x = x.astype("float64")

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Creating the RandomForestClassifier model
rf = RandomForestClassifier(random_state=2)

# Defining the parameter grid for GridSearchCV
param_grid = {"n_estimators": [10, 50, 100], "max_depth": [None, 10, 20, 30]}

# Applying GridSearchCV
grid_search = GridSearchCV(
    estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2
)

# Set up MLflow experiment
mlflow.set_experiment("breast-cancer-rf-hp")

with mlflow.start_run():
    # Fit the model
    grid_search.fit(x_train, y_train)

    # Get the best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Log parameters
    mlflow.log_params(best_params)

    # Log metrics - log as a dictionary
    mlflow.log_metrics({"accuracy": best_score})

    # Log training data (converting to a suitable format for MLflow)
    train_df = x_train.copy()
    train_df["target"] = y_train
    # Logging the training data as an artifact (instead of mlflow.data.from_pandas)
    train_df.to_csv("train_data.csv", index=False)
    mlflow.log_artifact("train_data.csv")

    # Log test data similarly
    test_df = x_test.copy()
    test_df["target"] = y_test
    test_df.to_csv("test_data.csv", index=False)
    mlflow.log_artifact("test_data.csv")

    # Log source code
    mlflow.log_artifact(__file__)

    # Log the best model with input example and signature
    # Prepare an example input (first row of the training data)
    input_example = x_train.iloc[0].to_dict()
    mlflow.sklearn.log_model(
        grid_search.best_estimator_,
        "random_forest",
        input_example=input_example,  # Providing an input example
    )

    # Set tags
    mlflow.set_tag("author", "Saloni Singh")

    # Print best parameters and score
    print(best_params)
    print(best_score)
