import numpy as np
from constants import *

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


def perform_classification(model, x):
    preds = []
    for classifier in model:
        preds.append(classifier.predict_(x))
    y_hat = np.zeros((x.shape[0], 1))
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


def translate_pred(preds):
    t_preds = []
    for pred in preds:
        if pred == RAVENCLAW:
            t_preds.append("Ravenclaw")
        elif pred == SLYTHERIN:
            t_preds.append("Slytherin")
        elif pred == GRYFFINDOR:
            t_preds.append("Gryffindor")
        elif pred == HUFFLEPUFF:
            t_preds.append("Hufflepuff")
    return t_preds