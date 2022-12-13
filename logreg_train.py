import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import sys
from my_logistic_regression import MyLogisticRegression
import sklearn.metrics
import pickle

COURSES = ["Astronomy", "Herbology", "Divination", "Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]
RAVENCLAW = 0
SLYTHERIN = 1
GRYFFINDOR = 2
HUFFLEPUFF = 3

def save_model(results):
    file = open('models.pickle', 'wb')
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

def normalize(x):
    norm_x = np.array([])
    for col in x.T:
        mean_col = np.mean(col)
        std_col = np.std(col)
        n_col = ((col - mean_col) / std_col).reshape((-1, 1))
        if norm_x.shape == (0,):
            norm_x = n_col
        else:
            norm_x = np.hstack((norm_x, n_col))
    return norm_x

def load_dataset():
    if len(sys.argv) != 2:
        raise Exception("Error : Wrong number of arguments -> Usage : python3 describe.py path/to/dataset.csv")
    df = pd.read_csv(sys.argv[1], index_col="Index")
    return df

def numerize(x):
    arr = np.copy(x)
    for i in range(len(arr)):
        if arr[i] == "Ravenclaw":
            arr[i] = RAVENCLAW
        elif arr[i] == "Slytherin":
            arr[i] = SLYTHERIN
        elif arr[i] == "Gryffindor":
            arr[i] = GRYFFINDOR
        elif arr[i] == "Hufflepuff":
            arr[i] = HUFFLEPUFF
    return arr


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

def perform_classification(model, x, y):
    preds = []
    for classifier in model:
        preds.append(classifier.predict_(x))
    y_hat = np.zeros(y.shape)
    for i, cl_zero_pred, cl_one_pred, cl_two_pred, cl_three_pred in zip(range(y_hat.shape[0]), preds[0], preds[1], preds[2], preds[3]):
        best = max(cl_zero_pred, cl_one_pred, cl_two_pred, cl_three_pred)
        if best == cl_zero_pred:
            y_hat[i] = RAVENCLAW
        elif best == cl_one_pred:
            y_hat[i] = SLYTHERIN
        elif best == cl_two_pred:
            y_hat[i] = GRYFFINDOR
        elif best == cl_three_pred:
            y_hat[i] = HUFFLEPUFF
    return y_hat

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
    preds = perform_classification(model, x, y)
    print(f'Precision : {accuracy_score_(y, preds)}')



def main():
    try:
        df = load_dataset()
    except Exception as e:
        print(e, file=sys.stderr)
        return
    df = clean_df(df)
    model = perform_one_vs_all(df)
    
    

if __name__ == "__main__":
    main()