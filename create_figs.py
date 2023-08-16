import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import os
import pandas as pd
from pathlib import Path
import pickle
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Union, Tuple, List

from har import Classifier, load_data, define_cv_folds

# font sizes
SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIG_SIZE = 19
BIGGER_SIZE = 25


def main():

    # setup fonts for matplotlib graphs
    set_fonts()

    # load training data
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

    # load trained classifiers
    clfs = load_clfs(Path("trained_clfs/"))

    # create graphics detailing confusion matrices
    form_confusion_matrices(
        clfs,
        df_train,
        "subject",
        features,
        target,
        target_labels,
    )

    # create 2d PCA visualization of data
    twod_pca_visual(
        df_train,
        features,
        target,
        target_labels,
    )


def twod_pca_visual(
    df_train: pd.DataFrame,
    features: List[str],
    target: str,
    target_labels: pd.DataFrame,
) -> None:
    """
    Create 2d representation of training data based on PCA
    :param df_train: training data
    :param features: list of feature column names
    :param target: target column name
    :param target_labels: df linking target number to english label
    :return: None (creates graph)
    """

    # setup PCA transformation
    scaler = StandardScaler()
    pca_2comp = PCA(n_components=2)
    pipe_pca = Pipeline(steps=[("scaler", scaler), ("pca", pca_2comp)])

    # setup map from target --> color, label
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
    ]
    target_map = dict()
    for target_val, target_label, color in zip(
        target_labels.iloc[:, 0], target_labels.iloc[:, 1], colors
    ):
        target_map[target_val] = {
            "label": target_label,
            "color": color,
        }

    # transform data and add target to df
    x = pipe_pca.fit_transform(df_train[features])
    df_pca = pd.DataFrame(x)
    df_pca = pd.concat([df_pca, df_train[target]], axis=1)
    df_pca.columns = ["first_pc", "second_pc", "target"]

    gridspec = {"height_ratios": [1.5, 5], "width_ratios": [5, 0.9]}

    # scatter plot by each activity
    n_pts = 250  # number of points of each activity to plot
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5), gridspec_kw=gridspec)

    for activity, dfg in df_pca.groupby("target"):

        # randomly sample n_pts from all observations of this activity
        index = np.random.choice(np.arange(dfg.shape[0]), n_pts, replace=False)

        # scatter plot x = 1st pc, y = 2nd pc
        ax[1, 0].scatter(
            dfg.first_pc.iloc[index],
            dfg.second_pc.iloc[index],
            c=target_map[activity]["color"],
            alpha=0.2,
            label=target_map[activity]["label"],
        )

        # histogram above of 1st pc values
        ax[0, 0].hist(
            dfg.first_pc.iloc[index],
            histtype="step",
            facecolor=target_map[activity]["color"],
            alpha=1,
            linewidth=2,
            label=target_map[activity]["label"],
        )

        # histogram to the right of 2nd pc values
        ax[1, 1].hist(
            dfg.second_pc.iloc[index],
            histtype="step",
            facecolor=target_map[activity]["color"],
            alpha=1,
            linewidth=2,
            orientation="horizontal",
            label=target_map[activity]["label"],
        )

    # hide ticks and axes for all plots except lower left
    minor_plots = [ax[1, 1], ax[0, 0], ax[0, 1]]
    for mp in minor_plots:
        plt.sca(mp)
        plt.xticks([])
        plt.yticks([])
        mp.axis("off")

    # add labels to lower left plot
    plt.sca(ax[1, 0])
    plt.xlabel(
        f"1st Principal Component\nProp. Var. Explained = {100 * pca_2comp.explained_variance_ratio_[0]:.1f}%"
    )
    plt.ylabel(
        f"2nd Principal Component\nProp. Var. Explained = {100 * pca_2comp.explained_variance_ratio_[1]:.1f}%"
    )

    leg = ax[1, 0].legend(loc=4, prop={"size": 10})
    for lh in leg.legendHandles:
        lh.set_alpha(1)  # make all legend entries opaque

    # Add title
    plt.sca(ax[0, 0])
    plt.title("PCA n=2 components", fontsize=16)

    # get rid of splace beteween subplots
    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig("figures/pca_2comp_hist")


def set_fonts():
    """
    setup default fonts for matplotlib graphs
    :return:
    """
    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def load_clfs(load_path: Path) -> List[Classifier]:
    """
    load trained classifiers
    :param load_path: directory where trained classifiers are stored
    :return: list of trained classifiers
    """

    clfs = []
    for f in os.listdir(load_path):
        abs_path = load_path / f
        clfs.append(pickle.load(open(abs_path, "rb")))

    return clfs


def form_confusion_matrices(
    clfs: List[Classifier],
    df_train: pd.DataFrame,
    grouping_col: str,
    features: List,
    target: str,
    target_labels: pd.DataFrame,
) -> None:
    """
    create graphics of confusion matrices
    :param clfs: list of classifiers
    :param df_train: training data
    :param grouping_col: name of column used to in establishing cv folds
    :param features: list of feature column names
    :param target: name of target column
    :param target_labels: labels for targets
    :return: None
    """

    for clf in clfs:
        # use best parameters
        est = clf.search_obj.estimator
        est.set_params(**clf.search_obj.best_params_)

        # setup cross validation folds
        df_train, _ = define_cv_folds(df_train, grouping_col, cross_val=True, nfolds=5)

        # pre-setup a confusion matrix that is initially set to all zeros
        cv_confusion_matrix = np.zeros(
            shape=(len(df_train[target].unique()), len(df_train[target].unique()))
        )

        # cycle through all folds for training/validation
        for val in df_train.fold.unique():
            validation_set = df_train[df_train.fold == val].copy()
            training_set = df_train[df_train.fold != val].copy()

            # retrain model
            est.fit(training_set[features], training_set[target])
            val_preds = est.predict(validation_set[features])

            # update confusion matrix (once per fold)
            cv_confusion_matrix = cv_confusion_matrix + confusion_matrix(
                validation_set[target].ravel(), val_preds
            )

        create_confusion_matrix_figure(
            cv_confusion_matrix, target_labels.iloc[:, 1].tolist(), clf
        )


def add_precision_and_recall_to_matrix(
    confusion: npt.NDArray,
    class_labels: list,
) -> pd.DataFrame:
    """
    Add precision and recall as last rows and columns of confusion matrix
    :param confusion: confusion matrix
    :param class_labels: class labels for target
    :return:
    """
    df_confusion = pd.DataFrame(confusion, columns=class_labels, index=class_labels)

    # calculate precision and recall
    precision = pd.Series(
        np.diag(df_confusion) / df_confusion.sum(axis=1), name="Precision"
    )
    recall = pd.Series(np.diag(df_confusion) / df_confusion.sum(axis=0), name="Recall")

    # add precision as last column and recall as bottom row
    df_confusion = pd.concat([df_confusion, precision], axis=1)
    df_confusion = pd.concat([df_confusion, pd.DataFrame(recall).transpose()], axis=0)

    return df_confusion


def create_graph_masks(n: int) -> tuple:
    """
    Create masks for confusion matrix (correct predictions, wrong predictions, precision/recall)
    :param n: confusion matrix is of size n x n (where the last row and col are precision and recall)
    :return: confusion matrix masks
    """

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


def create_confusion_matrix_figure(
    confusion: npt.NDArray,
    class_labels: list,
    clf: Classifier,
) -> None:
    """
    Create figures of confusion matrices
    :param confusion: confusion matrices
    :param class_labels: class labels
    :param clf: list of classifiers
    :return: None (saves figures locally)
    """

    # calculate CV accuracy
    accuracy = np.sum(np.eye(confusion.shape[0]) * confusion) / np.sum(confusion)

    # add precision and recall to confusion matrix
    df_confusion = add_precision_and_recall_to_matrix(confusion, class_labels)

    # create boolean masks that will be used for graphing
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
        fmt=".0f",
    )

    # plot wrong predictions (off-diagonal elements)
    sns.heatmap(
        df_confusion,
        cbar=False,
        annot=True,
        mask=wrong_pred_mask,
        cmap="OrRd",
        fmt=".0f",
    )

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

    plt.title(clf.full_name, fontsize=BIG_SIZE)
    plt.tight_layout()

    fig_path = Path("figures/")
    fig_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path / f"{clf.short_name}_confusion.png")


if __name__ == "__main__":
    main()
