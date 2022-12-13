import numpy as np
import pandas as pd
import sys
from my_logistic_regression import MyLogisticRegression
import pickle
from constants import *
from utils import *


def save_model(results):
    file = open('model.pickle', 'wb')
    pickle.dump(results, file)
    file.close()


def accuracy_score_(y, y_hat):
    try:
        if type(y) != np.ndarray or type(y_hat) != np.ndarray or y.shape != y_hat.shape:
            return None
        t = 0
        for y_i, y_hat_i in zip(y, y_hat):
            if y_i == y_hat_i:
                t += 1
        if len(y) == 0:
            return 0
        return t / len(y)
    except:
        return None



def load_dataset():
    if len(sys.argv) != 2:
        raise Exception("Error : Wrong number of arguments -> Usage : python3 describe.py path/to/dataset.csv")
    df = pd.read_csv(sys.argv[1], index_col="Index")
    return df


def binarize(x, reference):
    arr = np.copy(x)
    for i in range(arr.shape[0]):
        if reference == float(arr[i]):
            arr[i] = 1
        else:
            arr[i] = 0
    return arr


def train_classifier(x_train, y_train, reference):
    myLR = MyLogisticRegression(theta=np.random.rand(x_train.shape[1] + 1, 1).reshape(-1, 1), max_iter=5000)
    y_train = binarize(y_train, reference)
    print("start fitting")
    myLR.fit_(x_train, y_train)
    print("fiting done")
    return myLR


def clean_df(df):
    return df[~df.isnull().any(axis=1)]


def perform_one_vs_all(df):
    x = df[COURSES].to_numpy()
    y = df["Hogwarts House"]
    y  = numerize(y).reshape(-1, 1)
    x = normalize(x)
    model = []
    model.append(train_classifier(x, y, RAVENCLAW))
    model.append(train_classifier(x, y, SLYTHERIN))
    model.append(train_classifier(x, y, GRYFFINDOR))
    model.append(train_classifier(x, y, HUFFLEPUFF))
    preds = perform_classification(model, x)
    return model


def main():
    try:
        df = load_dataset()
    except Exception as e:
        print(e, file=sys.stderr)
        return
    df = clean_df(df)
    model = perform_one_vs_all(df)
    save_model(model)
    
    

if __name__ == "__main__":
    main()