import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import PredefinedSplit
from typing import Union, Tuple


def main():
    pass
    df_train, target, features = load_data(
        subject_path=Path("uci_har_dataset/train/subject_train.txt"),
        features_path=Path("uci_har_dataset/features.txt"),
        x_path=Path("uci_har_dataset/train/X_train.txt"),
        y_path=Path("uci_har_dataset/train/y_train.txt"),
    )
    df_test, _, _ = load_data(
        subject_path=Path("uci_har_dataset/test/subject_test.txt"),
        features_path=Path("uci_har_dataset/features.txt"),
        x_path=Path("uci_har_dataset/test/X_test.txt"),
        y_path=Path("uci_har_dataset/test/y_test.txt"),
    )

    df_train, ps = define_training_folds(
        df_train,
        grouping_col="subject",
        cross_val=False,
    )


def load_data(
    subject_path: Path,
    features_path: Path,
    x_path: Path,
    y_path: Path,
) -> Tuple[pd.DataFrame, str, list]:

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

    # load subjects (the person that the test is being performed on)
    subjects = pd.read_csv(subject_path, delim_whitespace=True, header=None)

    # combine target, subject, and features, and rename columns
    df = pd.concat([y, subjects, x], axis=1, ignore_index=True)
    df.columns = [target, "subject"] + feature_names

    return df, target, feature_names


def define_training_folds(
    df: pd.DataFrame,
    grouping_col: str,
    cross_val: bool = True,
    nfolds: int = 5,
    training_frac: float = 0.8,
    random_state: Union[int, None] = None,
) -> Tuple[pd.DataFrame, PredefinedSplit]:
    if random_state is not None:
        np.random.seed(random_state)

    if cross_val:
        # form dict mapping grouping column (subject id) to their cross-validation groupf
        folds = dict(
            zip(
                np.random.permutation(df[grouping_col].unique()),
                np.repeat(
                    np.arange(nfolds), (len(df[grouping_col].unique()) // nfolds) + 1
                ),
            )
        )
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


if __name__ == "__main__":
    main()
