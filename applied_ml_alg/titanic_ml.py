import joblib
import logging
import logging.config
import os
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from time import time
import warnings
import yaml

from load_data import load_titanic_data

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def print_results(results: dict):
    """Print the results of ml fit.

    Paramters:
        results: The results of the ml fit.
    """
    print(f"Best params: {results.best_params_}\n")
    means = results.cv_results_["mean_test_score"]
    stds = results.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, results.cv_results_["params"]):
        mean = round(mean, 3)
        std = round(std * 2, 3)
        print(f"{mean} (+/- {std}) for {params}")


def run_logistic_regression(ml_ds: dict) -> object:
    """Runs the logistic regression and fit.

    When to use it:
    - Bianry target variable.
    - Transparency is important or interested in significance of predictors.
    - Fairly well behaved data (not too many outliers or missing values).
    - Need a quick initial benchmark.

    When not to use it:
    - Continuous target variable.
    - Massive data (rows or columns).
    - Unwildy data (outliers, missing values, skewed features or complex
        relationshiops).
    - Performance is the only thing that matters.

    Parameters:
        ml_ds: The data dict.

    Returns:
        cv object.
    """
    LOGGER.info("Running logistic regression.")
    lr = LogisticRegression()
    parameters = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    # cv is a cross validatiion generator
    cv = GridSearchCV(estimator=lr, param_grid=parameters, cv=5)
    cv.fit(ml_ds["features"], ml_ds["labels"])
    print_results(cv)
    print(cv.best_estimator_)
    return cv


def run_svm(ml_ds: dict) -> object:
    """A support vector machine (svm) is a classifier that fines an optimal
    boundary between two groups of data points. It is a line midway between the
    groups.

    When to use it?
    - Bianary target variable.
    - Feature to row ratio is very high.
    - Very complex relationships.
    - Lots of outliers.

    When not to use it?
    - Feature to row ratio is very low.
    - Tranparency is important or interested in significance of predictors.
    - Looking for a quick benchmark model.

    Parameters:
        ml_ds: The data dict.

    Returns:
        cv object.
    """
    LOGGER.info("Running support vector machine (svm).")
    svc = SVC()
    # rbf - radial basis kernel; this determines the relationship between
    # points in infinite dimensions to ultimately group them together.
    parameters = {
        "kernel": ["linear", "rbf"],
        "C": [0.1, 1, 10],
    }
    cv = GridSearchCV(estimator=svc, param_grid=parameters, cv=5)
    cv.fit(ml_ds["features"], ml_ds["labels"])
    print_results(cv)
    return cv


def run_mlp(ml_ds: dict) -> object:
    """Multilayer perceptron is a classic feed-forward artificial neural
    network, the core component of deep learning. A connected series of nodes
    (in the form of a directed acyclic graph), where each node represents a
    function or a model.

    When to use it?
    - Categorical or continuous target variable.
    - Very complex relationships or performance is the only thing that matters.
    - When control over the training process is very important.

    When not to use it?
    - Image recognition, time series, etc.
    - Tranparency is important or interested in significance of predictors.
    - Need a quick benchmark model.
    - Limited data available.

    Parameters:
        ml_ds: The data dict.

    Returns:
        cv object.
    """
    LOGGER.info("Running multilayer perceptron.")
    parameters = {}
    # Hidden layer-size hyperparameter determines how many hidden layers there
    # will be an dhow many nodes in each layer. Using 1 hidden layer because it
    # is a simple problem. (nodes, layers)
    parameters["hidden_layer_sizes"] = [(10, 1), (50, 1), (100, 1)]

    # Activation function hyperparameter dictates the type of nonlinearity that
    # is introduced to the model: sigmoid, TanH (hyperbolic tangent), ReLU
    # (Rectified linear unit - sets all negative numbers to 0).
    parameters["activation"] = ["relu", "tanh", "logistic"]

    # Learning rate hyperparameter facilitates both how quickly and whether or
    # not the algorithm will find the optimal solution.
    # Constant - keep rate the same during learning process.
    # Invscaling - gradually decrease learning rate at each step.
    # Adaptive - keep the rate constant as long as the training loss keeps
    #   decreasing. If the learning rate stops going down, then it'll decrease
    #   the learning rate so it takes smaller steps.
    parameters["learning_rate"] = ["constant", "invscaling", "adaptive"]

    mlp = MLPClassifier()
    cv = GridSearchCV(estimator=mlp, param_grid=parameters, cv=5)
    cv.fit(ml_ds["features"], ml_ds["labels"])
    print_results(cv)
    return cv


def run_random_forest(ml_ds: dict) -> object:
    """Random forest merges a collection of independent decision trees to get a
    more accurate and stable prediction. It is a type of ensemble method, which
    combines several machine learning models in order to decrease both bias and
    variance.

    When to use it?
    - Categorical or continuous target variable.
    - Interested in significance of predictors.
    - Need a quick benchmark model.
    - If you have messy data, such as missing values, outliers.

    When not to use it?
    - If you're solving a very complex, novel problem.
    - Transparency is important.
    - Prediction time is important.

    Parameters:
        ml_ds: The data dict.

    Returns:
        cv object.
    """
    LOGGER.info("Running random forest.")
    parameters = {}
    # N estimators hyperparameter controls how many individual decision trees
    # will be built. Width of trees.
    parameters["n_estimators"] = [5, 50, 250]

    # Max depth hyperparameter controls how deep each individual decision tree
    # can go. If this was too high, you would get a tree that has a node for
    # every example in the training set. Depth of trees.
    parameters["max_depth"] = [2, 4, 8, 16, 32, None]
    # cv is a cross validatiion generator
    rf = RandomForestClassifier()
    cv = GridSearchCV(estimator=rf, param_grid=parameters, cv=5)
    cv.fit(ml_ds["features"], ml_ds["labels"])
    print_results(cv)
    print(cv.best_estimator_)
    return cv


def run_boosting(ml_ds: dict) -> object:
    """Boosting is an ensemble method that aggregates a number of weak models
    to create one strong model. A weak model is one that is only slightly
    better than random guessing. A strong model is one that is strongly
    correlated with the true classification. Boosting effectively learns from
    its mistakes with each iteration.

    This is gradient boosted trees.

    When to use it?
    - Categorical or continuous target variable.
    - Useful on nearly any type of problem.
    - Interested in significance of predictors.
    - Prediction time is important.

    When not to use it?
    - Transparency is important.
    - Training time is important or compute power is limited.
    - Data is really noisy.

    Parameters:
        ml_ds: The data dict.

    Returns:
        cv object.
    """
    LOGGER.info("Running boosting.")
    parameters = {}
    # N estimators hyperparameter controls how many individual decision trees
    # will be built. Width of trees.
    parameters["n_estimators"] = [5, 50, 250, 500]

    # Max depth hyperparameter controls how deep each individual decision tree
    # can go. If this was too high, you would get a tree that has a node for
    # every example in the training set. Depth of trees.
    parameters["max_depth"] = [1, 3, 5, 7, 9]

    # Learning rate hyperparameter facilitates both how quickly and whether or
    # not the algorithm will find the optimal solution.
    parameters["learning_rate"] = [0.01, 0.1, 1, 10, 100]

    gb = GradientBoostingClassifier()
    cv = GridSearchCV(estimator=gb, param_grid=parameters, cv=5)
    cv.fit(ml_ds["features"], ml_ds["labels"])
    print_results(cv)
    print(cv.best_estimator_)
    return cv


def run_models(run_dir: object, ml_ds: dict):
    """Run the different machine learning models and save them.

    Parameters:
        run_dir: The path to the run directory.
        ml_ds: The data dict.
    """
    model_fn = run_dir.joinpath("outputs", "lr_model.pkl")
    if not model_fn.exists():
        cv = run_logistic_regression(ml_ds)
        joblib.dump(cv.best_estimator_, model_fn)

    model_fn = run_dir.joinpath("outputs", "svm_model.pkl")
    if not model_fn.exists():
        cv = run_svm(ml_ds)
        joblib.dump(cv.best_estimator_, model_fn)

    model_fn = run_dir.joinpath("outputs", "mlp_model.pkl")
    if not model_fn.exists():
        cv = run_mlp(ml_ds)
        joblib.dump(cv.best_estimator_, model_fn)

    model_fn = run_dir.joinpath("outputs", "rf_model.pkl")
    if not model_fn.exists():
        cv = run_random_forest(ml_ds)
        joblib.dump(cv.best_estimator_, model_fn)

    model_fn = run_dir.joinpath("outputs", "gb_model.pkl")
    if not model_fn.exists():
        cv = run_boosting(ml_ds)
        joblib.dump(cv.best_estimator_, model_fn)


def read_models(run_dir: object):
    """Read saved models.

    Parameters:
        run_dir: The path to the run directory.
    """
    models = {}
    saved_models = ("lr", "svm", "mlp", "rf", "gb")
    for model in saved_models:
        filename = run_dir.joinpath(f"{model}_model.pkl")
        models[model] = joblib.load(filename)

    return


def main():
    """The titanic machine learning problem. """
    pd.set_option("display.max_columns", None)
    run_dir = Path(RUN_PATH)
    ml_ds = load_titanic_data(run_dir)
    run_models(run_dir, ml_ds)
    read_models(run_dir)


if __name__ == "__main__":
    RUN_PATH = os.path.dirname(os.path.realpath(__file__))
    LOG_CONFIG = os.path.join(RUN_PATH, "log_config.yaml")
    with open(LOG_CONFIG, "r") as log_file:
        logging.config.dictConfig(yaml.safe_load(log_file.read()))

    LOGGER = logging.getLogger(__name__)
    main()
    logging.shutdown()
