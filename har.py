from collections import namedtuple
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from typing import Union, Tuple, List


Classifier = namedtuple(
    "Classifier",
    ["full_name", "short_name", "search_obj"],
)


def main():
    # load training data & names of target and features
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
    clfs = form_classifiers(ps=ps, random_state=123, limited_params=True)

    # train classifiers
    clfs = train_classifiers(
        df_train=df_train,
        features=features,
        target=target,
        clfs=clfs,
    )

    form_confusion_matrices(
        clfs,
        df_train,
        "subject",
        features,
        target,
        target_labels,
    )


def form_confusion_matrices(
    clfs, df_train, grouping_col, features, target, target_labels
):
    for clf in clfs:
        est = clf.search_obj.estimator
        est.set_params(**clf.search_obj.best_params_)

        df_train, _ = define_cv_folds(df_train, grouping_col, cross_val=True, nfolds=5)

        cv_confusion_matrix = np.zeros(
            shape=(len(df_train[target].unique()), len(df_train[target].unique()))
        )

        for fold in df_train[grouping_col].unique():
            validation_set = df_train[df_train[grouping_col] == fold].copy()
            training_set = df_train[df_train[grouping_col] != fold].copy()

            est.fit(training_set[features], training_set[target])
            val_preds = est.predict(validation_set[features])

            cv_confusion_matrix = cv_confusion_matrix + confusion_matrix(
                validation_set[target].ravel(), val_preds
            )

        create_confusion_maxtrix_figure(
            cv_confusion_matrix, target_labels.iloc[:, 1].tolist(), clf
        )


def add_precision_and_recall_to_matrix(confusion, class_labels):
    df_confusion = pd.DataFrame(confusion, columns=class_labels, index=class_labels)

    precision = pd.Series(
        np.diag(df_confusion) / df_confusion.sum(axis=1), name="Precision"
    )
    recall = pd.Series(np.diag(df_confusion) / df_confusion.sum(axis=0), name="Recall")

    df_confusion = pd.concat([df_confusion, precision], axis=1)
    df_confusion = pd.concat([df_confusion, pd.DataFrame(recall).transpose()], axis=0)

    return df_confusion


def create_graph_masks(n):

    # create boolean mask so that bad prediction have different color scheme than good predictions
    # note: mask applies to True values (not False)
    wrong_pred_mask = np.eye(n, dtype=bool)
    correct_pred_mask = ~wrong_pred_mask

    # last row and column (precision and recall) also need to be masked
    correct_pred_mask[n - 1, :] = True
    correct_pred_mask[:, n - 1] = True
    wrong_pred_mask[n - 1, :] = True
    wrong_pred_mask[:, n - 1] = True

    # create boolean masks for different color scheme for precision and recall
    prec_rec_mask = np.ones((n, n), dtype=bool)
    prec_rec_mask[n - 1, :] = False
    prec_rec_mask[:, n - 1] = False

    return wrong_pred_mask, correct_pred_mask, prec_rec_mask


def create_confusion_maxtrix_figure(
    confusion: npt.NDArray,
    class_labels: list,
    clf: Classifier,
) -> None:

    accuracy = np.sum(np.eye(confusion.shape[0]) * confusion) / np.sum(confusion)
    df_confusion = add_precision_and_recall_to_matrix(confusion, class_labels)
    wrong_pred_mask, correct_pred_mask, prec_rec_mask = create_graph_masks(
        df_confusion.shape[0]
    )

    fig = plt.figure(figsize=(8, 8))
    fig.add_subplot(1, 1, 1)

    # plot correct predictions (diagonal elements)
    sns.heatmap(
        df_confusion,
        mask=correct_pred_mask,
        cbar=False,
        annot=True,
        cmap=matplotlib.colors.ListedColormap(["tab:blue"]),
    )

    # plot wrong predictions (off-diagonal elements)
    sns.heatmap(df_confusion, cbar=False, annot=True, mask=wrong_pred_mask, cmap="OrRd")

    # plot precision and recall (last row and last col)
    sns.heatmap(
        df_confusion,
        cbar=False,
        annot=True,
        mask=prec_rec_mask,
        cmap="RdYlGn",
        fmt=".1%",
    )

    plt.ylabel("True Class")
    plt.xlabel(f"\n\nCV Accuracy = {100 * accuracy:.2f}%")

    # move class labels to top of graph
    plt.tick_params(
        axis="x",
        which="major",
        bottom=False,
        top=False,
        labelbottom=False,
        labeltop=True,
        labelrotation=90,
    )

    plt.title(clf.full_name)
    plt.tight_layout()

    fig_path = Path("figures/")
    fig_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path / f"{clf.short_name}_confusion.png")


def load_data(
    subject_path: Path,
    features_path: Path,
    x_path: Path,
    y_path: Path,
    y_label_path: Path,
) -> Tuple[pd.DataFrame, str, list, pd.DataFrame]:

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

    # form dict of true names of the targets
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

    # assign random seed if specified
    if random_state is not None:
        np.random.seed(random_state)

    if cross_val:  # cross-validation
        # form dict mapping grouping column (subject id) to their cross-validation group
        folds = dict(
            zip(
                np.random.permutation(df[grouping_col].unique()),
                np.repeat(np.arange(nfolds), len(df[grouping_col].unique())),
            )
        )
    else:  # separate validation set
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

    clfs = []

    clf_setup_fns = [pca_lr, lr, pca_lda, lda, svm, random_forest]

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
    svm = SVC()

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
            estimator=svm,
            param_grid=params,
            scoring="accuracy",
            cv=ps,
            return_train_score=True,
            verbose=4,
        ),
    )


def random_forest(ps, random_state, limited_params):

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

    for clf in clfs:
        clf.search_obj.fit(df_train[features], df_train[target].ravel())

        print(clf.full_name)
        print(f"Best Parameters: {clf.search_obj.best_params_}")
        print(f"Top CV score: {clf.search_obj.best_score_}\n\n")

    return clfs


if __name__ == "__main__":
    main()
