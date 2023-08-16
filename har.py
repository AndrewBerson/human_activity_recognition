from collections import namedtuple
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from typing import Union, Tuple, List


# Named tuple to hold each classifier
Classifier = namedtuple(
    "Classifier",
    ["full_name", "short_name", "search_obj"],
)


def main():
    # load training data, name of target column, name of feature cols,
    # and df linking targets to their english labels (as opposed to numbers)
    df_train, target, features, target_labels = load_data(
        subject_path=Path("uci_har_dataset/train/subject_train.txt"),
        features_path=Path("uci_har_dataset/features.txt"),
        x_path=Path("uci_har_dataset/train/X_train.txt"),
        y_path=Path("uci_har_dataset/train/y_train.txt"),
        y_label_path=Path("uci_har_dataset/activity_labels.txt"),
    )

    # load test data
    df_test, _, _, _ = load_data(
        subject_path=Path("uci_har_dataset/test/subject_test.txt"),
        features_path=Path("uci_har_dataset/features.txt"),
        x_path=Path("uci_har_dataset/test/X_test.txt"),
        y_path=Path("uci_har_dataset/test/y_test.txt"),
        y_label_path=Path("uci_har_dataset/activity_labels.txt"),
    )

    # form splits for cross-validation or for separate validation set
    df_train, ps = define_cv_folds(
        df=df_train,
        grouping_col="subject",
        cross_val=False,
    )

    # form classifiers
    clfs = form_classifiers(ps=ps, random_state=None, limited_params=False)

    # train classifiers
    clfs = train_classifiers(
        df_train=df_train,
        features=features,
        target=target,
        clfs=clfs,
    )

    # save trained classifiers locally
    dump_classifiers(clfs, Path("trained_clfs"))

    # choose best classifier from best CV score and evaluate test accuracy
    choose_clf_and_eval_test_score(clfs, df_test, target, features)


def choose_clf_and_eval_test_score(
    clfs: List[Classifier],
    df_test: pd.DataFrame,
    target: str,
    features: List[str],
) -> None:
    """
    Choose best classifier based on best CV score and evaluate test accuracy
    :param clfs: list of Classifiers
    :param df_test: test data
    :param target: name of target column
    :param features: name of feature columns
    :return: None
    """

    # find best classifier
    best_clf = clfs[0]
    best_score = clfs[0].search_obj.best_score_
    for clf in clfs[1:]:
        if best_score < clf.search_obj.best_score_:
            best_clf = clf
            best_score = clf.search_obj.best_score_

    # evaluate test accuracy for best classifier
    test_preds = best_clf.search_obj.best_estimator_.predict(df_test[features])
    test_acc = accuracy_score(df_test[target].ravel(), test_preds)

    # report results
    print(f"Best model: {best_clf.full_name}")
    print(f"Best parameters: {best_clf.search_obj.best_params_}")
    print(f"Test accuracy = {test_acc * 100:.2f}%")


def load_data(
    subject_path: Path,
    features_path: Path,
    x_path: Path,
    y_path: Path,
    y_label_path: Path,
) -> Tuple[pd.DataFrame, str, list, pd.DataFrame]:
    """
    Create dataframe of train or test data
    :param subject_path: path to txt file containing which (human) subject the test is being performed on
    :param features_path: path to txt file outlining the names of all of the features
    :param x_path: path to txt file containing feature measurements
    :param y_path: path to txt file containing targets
    :param y_label_path: path to txt file that maps the target number into english
    :return: tuple of (formatted data in DataFrame, target column name, [feature col names], target labels)
    """

    # load feature names
    feature_names = (
        pd.read_csv(features_path, delim_whitespace=True, header=None)
        .iloc[:, 1]
        .tolist()
    )

    # rename duplicated feature names
    new_feature_names = []
    for i, feature in enumerate(feature_names):
        new_feature_names.append(f"{feature}_{feature_names[:i].count(feature)}")
    feature_names = new_feature_names

    # load features
    x = pd.read_csv(x_path, delim_whitespace=True, header=None)

    # load targets
    y = pd.read_csv(y_path, delim_whitespace=True, header=None)
    target = "target"

    # load target labels
    target_labels = pd.read_csv(y_label_path, delim_whitespace=True, header=None)

    # load subjects (the person that the test is being performed on)
    subjects = pd.read_csv(subject_path, delim_whitespace=True, header=None)

    # combine target, subject, and features, and rename columns
    df = pd.concat([y, subjects, x], axis=1, ignore_index=True)
    df.columns = [target, "subject"] + feature_names

    return df, target, feature_names, target_labels


def define_cv_folds(
    df: pd.DataFrame,
    grouping_col: str,
    cross_val: bool = True,
    nfolds: int = 5,
    training_frac: float = 0.8,
    random_state: Union[int, None] = None,
) -> Tuple[pd.DataFrame, PredefinedSplit]:
    """
    Function used to define folds for cross validation or validation set
    :param df: training data
    :param grouping_col: column used to ensure that different CV folds do not share any values within this column.
        Tests were performed on humans, so it's best to keep each test participant in a different CV group.
    :param cross_val: Bool - whether or not to use cross-validation or separate validation set
    :param nfolds: number of folds for cross validation (only applicable if cross_val == True)
    :param training_frac: Fraction of data to use for training. Rest is used for Validation set. Only applicable
        if cross_val == False.
    :param random_state: random seed
    :return: Training data and predefined split for cross validation (Note: adds a column "fold" to df)
    """

    # assign random seed if specified
    if random_state is not None:
        np.random.seed(random_state)

    # cross validation
    if cross_val:
        # form dict mapping grouping column (subject id) to their cross-validation group
        folds = dict(
            zip(
                np.random.permutation(df[grouping_col].unique()),
                list(np.arange(nfolds)) * len(df[grouping_col].unique()),
            )
        )

    # separate validation set
    else:
        # form dict matching grouping column (subject id) to training set (-1) or validation set (0)
        folds = dict(
            zip(
                np.random.permutation(df[grouping_col].unique()),
                [-1] * int(training_frac * len(df[grouping_col].unique()))
                + [0] * len(df[grouping_col].unique()),
            )
        )

    # add fold to df and form predefined split
    df["fold"] = df[grouping_col].map(folds)
    ps = PredefinedSplit(df["fold"])

    return df, ps


def form_classifiers(
    ps: PredefinedSplit,
    random_state: Union[int, None] = None,
    limited_params: bool = True,
) -> List[Classifier]:
    """
    Function to form list of Classifiers
    :param ps: predefined split for cross-validation
    :param random_state: random stated
    :param limited_params: Whether or not to train with limited parameter to increase speed of training
    :return: list of Classifiers
    """

    # setup list of functions that return a single Classifier
    clf_setup_fns = [pca_lr, lr, pca_lda, lda, svm, random_forest]
    # clf_setup_fns = [pca_lr, lr]

    # build list of classifiers
    clfs = []
    for fn in clf_setup_fns:
        clfs.append(
            fn(
                ps=ps,
                random_state=random_state,
                limited_params=limited_params,
            )
        )

    return clfs


def pca_lr(ps, random_state, limited_params):
    """
    Form classifier - principal component analysis followed by logistic regression
    :param ps: predefined split for cross-validation or for separate validation set
    :param random_state: random seed
    :param limited_params: whether or not to limit the number of parameters to speed up training time
    :return: Classifier
    """

    # PCA followed by logistic regression
    scaler = StandardScaler()
    pca = PCA()
    logistic = LogisticRegression(
        random_state=random_state,
        max_iter=100,
        multi_class="multinomial",
        solver="saga",
        tol=1e-4,
    )
    pipe_pca_lr = Pipeline(
        steps=[("scaler", scaler), ("pca", pca), ("logistic", logistic)]
    )

    if limited_params:
        params = {
            "pca__n_components": [0.8, 0.975],
            "logistic__C": np.logspace(-4, 4, 2),
        }
    else:
        params = {
            "pca__n_components": [0.6, 0.7, 0.8, 0.9, 0.925, 0.95, 0.975, 0.99],
            "logistic__C": np.logspace(-4, 4, 20),
        }

    return Classifier(
        "PCA & Normalized Logistic Regression",
        "pca_lr",
        GridSearchCV(
            estimator=pipe_pca_lr,
            param_grid=params,
            scoring="accuracy",
            cv=ps,
            return_train_score=True,
            verbose=4,
        ),
    )


def lr(ps, random_state, limited_params):
    """
    Form classifier - logistic regression
    :param ps: predefined split for cross-validation or for separate validation set
    :param random_state: random seed
    :param limited_params: whether or not to limit the number of parameters to speed up training time
    :return: Classifier
    """

    # Logistic regression
    logistic = LogisticRegression(
        random_state=random_state,
        max_iter=100,
        multi_class="multinomial",
        solver="saga",
        tol=1e-4,
    )

    pipe_lr = Pipeline(steps=[("logistic", logistic)])

    if limited_params:
        params = {
            "logistic__C": np.logspace(-4, 4, 2),
        }
    else:
        params = {
            "logistic__C": np.logspace(-4, 4, 20),
        }

    return Classifier(
        "Normalized Logistic Regression",
        "lr",
        GridSearchCV(
            estimator=pipe_lr,
            param_grid=params,
            scoring="accuracy",
            cv=ps,
            return_train_score=True,
            verbose=4,
        ),
    )


def pca_lda(ps, random_state, limited_params):
    """
    Form classifier - principal component analysis followed by linear discriminant analysis
    :param ps: predefined split for cross-validation or for separate validation set
    :param random_state: random seed
    :param limited_params: whether or not to limit the number of parameters to speed up training time
    :return: Classifier
    """

    # PCA followed by LDA
    scaler = StandardScaler()
    pca = PCA()
    lda = LinearDiscriminantAnalysis(solver="eigen")
    pipe_pca_lda = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("lda", lda)])

    if limited_params:
        params = {
            "pca__n_components": [0.6, 0.975],
            "lda__shrinkage": np.linspace(0, 1, 2),
        }
    else:
        params = {
            "pca__n_components": [0.6, 0.7, 0.8, 0.9, 0.925, 0.95, 0.975, 0.99],
            "lda__shrinkage": np.linspace(0, 1, 5),
        }

    return Classifier(
        "PCA & LDA",
        "pca_lda",
        GridSearchCV(
            estimator=pipe_pca_lda,
            param_grid=params,
            scoring="accuracy",
            cv=ps,
            return_train_score=True,
            verbose=4,
        ),
    )


def lda(ps, random_state, limited_params):
    """
    Form classifier - linear discriminant analysis
    :param ps: predefined split for cross-validation or for separate validation set
    :param random_state: random seed
    :param limited_params: whether or not to limit the number of parameters to speed up training time
    :return: Classifier
    """

    lda = LinearDiscriminantAnalysis(solver="eigen")
    pipe_lda = Pipeline(steps=[("lda", lda)])

    if limited_params:
        params = {"lda__shrinkage": np.linspace(0, 1, 2)}
    else:
        params = {"lda__shrinkage": np.linspace(0, 1, 5)}

    return Classifier(
        "LDA",
        "lda",
        GridSearchCV(
            estimator=pipe_lda,
            param_grid=params,
            scoring="accuracy",
            cv=ps,
            return_train_score=True,
            verbose=4,
        ),
    )


def svm(ps, random_state, limited_params):
    """
    Form classifier - support vector machine
    :param ps: predefined split for cross-validation or for separate validation set
    :param random_state: random seed
    :param limited_params: whether or not to limit the number of parameters to speed up training time
    :return: Classifier
    """

    svm_clf = SVC(random_state=random_state)

    if limited_params:
        params = {
            "C": np.logspace(-4, 4, 2),
        }
    else:
        params = {
            "C": np.logspace(-4, 4, 10),
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
        }

    return Classifier(
        "Support Vector Machine",
        "svm",
        GridSearchCV(
            estimator=svm_clf,
            param_grid=params,
            scoring="accuracy",
            cv=ps,
            return_train_score=True,
            verbose=4,
        ),
    )


def random_forest(ps, random_state, limited_params):
    """
    Form classifier - random forest
    :param ps: predefined split for cross-validation or for separate validation set
    :param random_state: random seed
    :param limited_params: whether or not to limit the number of parameters to speed up training time
    :return: Classifier
    """

    rf = RandomForestClassifier(n_jobs=1, random_state=random_state, n_estimators=100)

    if limited_params:
        params = {
            "max_depth": [10],
            "min_samples_leaf": [8],
            "max_features": ["sqrt", None],
        }
    else:
        params = {
            "max_depth": [3, 5, 10, 20, 40],
            "min_samples_leaf": [2, 5, 8, 10, 20],
            "max_features": ["sqrt", "log2", None],
        }

    return Classifier(
        "Random Forest",
        "rf",
        GridSearchCV(
            estimator=rf,
            param_grid=params,
            scoring="accuracy",
            cv=ps,
            return_train_score=True,
            verbose=4,
        ),
    )


def train_classifiers(
    df_train: pd.DataFrame,
    features: List,
    target: str,
    clfs: List[Classifier],
) -> List[GridSearchCV]:
    """
    Train classifiers on training data
    :param df_train: training data
    :param features: feature names
    :param target: target name
    :param clfs: list of classifiers
    :return: list of classifers that have been trained
    """

    for clf in clfs:
        clf.search_obj.fit(df_train[features], df_train[target].ravel())

    return clfs


def dump_classifiers(
    clfs: List[Classifier],
    dump_dir: Path,
) -> None:
    """
    Save classifiers to local machine
    :param clfs: list of classifers
    :param dump_dir: where to save the classifiers
    :return: None (saves classifiers locally to dump_dir
    """

    dump_dir.mkdir(parents=True, exist_ok=True)

    for clf in clfs:
        abs_path = dump_dir / f"{clf.short_name}.pkl"

        with open(str(abs_path), "wb") as f:
            pickle.dump(clf, f)


if __name__ == "__main__":
    main()
