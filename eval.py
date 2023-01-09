import pandas as pd
import sys
from my_logistic_regression import MyLogisticRegression
from constants import *
from utils import *
from sklearn.linear_model import LogisticRegression
import sklearn

def load_datasets():
    if len(sys.argv) != 4:
        raise Exception("Error : Wrong number of arguments -> Usage : python3 describe.py path/to/train_dataset.csv path/to/test_dataset.csv path/to/preds.csv")
    train_df = pd.read_csv(sys.argv[1], index_col="Index")
    test_df = pd.read_csv(sys.argv[2], index_col="Index")
    my_preds = pd.read_csv(sys.argv[3], index_col="Index")
    return train_df, test_df, my_preds


def clean_df(df):
    return df[~df.isnull().any(axis=1)]


def train_eval_model(df):
    df = clean_df(df)
    x = df[COURSES]
    y = df["Hogwarts House"]
    lr = LogisticRegression(max_iter=5000)
    lr.fit(x, y)
    return lr

def get_eval_predictions(model, df):
    xdf = df[COURSES]
    x = clean_df(xdf)
    preds = model.predict(x)
    return preds

def main():
    try:
        train_df, test_df, my_preds = load_datasets()
    except Exception as e:
        print(e, file=sys.stderr)
        return
    eval_model = train_eval_model(train_df)
    preds = get_eval_predictions(eval_model, test_df)
    print(sklearn.metrics.accuracy_score(preds, my_preds))
    

if __name__ == "__main__":
    main()