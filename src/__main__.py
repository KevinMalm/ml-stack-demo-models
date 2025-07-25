import os
import mlflow
import json
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    mlflow_configure()

    with mlflow.start_run():
        # Load data
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

        # Train model
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log parameters, metrics, and model
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(clf, "model")

        print(f"Run complete. Accuracy: {acc}")


def mlflow_configure():
    # Load Env Vars
    _experiment = os.getenv("MLFLOW_EXPERIMENT_NAME")
    _tags = os.getenv("MLFLOW_EXPERIMENT_TAGS")
    # Validate
    if _experiment is None or _tags is None:
        raise Exception("Environment has not been setup correctly.")
    # Parse the ML Flow TAGs
    print(_tags)
    tags = json.loads(_tags.replace(r"\\", ""))

    # Set the Experiment name
    mlflow.set_experiment(_experiment)
    # Set the Experiment tags
    for x in tags:
        mlflow.set_tag(x["key"], x["value"])


if __name__ == "__main__":
    main()
