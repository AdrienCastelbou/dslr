import pandas as pd
import sys
import pickle
from constants import *
from utils import *

def load_files():
    if len(sys.argv) != 3:
        raise Exception("Error : Wrong number of arguments -> Usage : python3 logreg_predict.py path/to/dataset.csv path/to/model.pickle")
    df = pd.read_csv(sys.argv[1], index_col="Index")
    file = open(sys.argv[2], 'rb')
    model = pickle.load(file)
    return df, model

def save_preds(preds):
    pd.DataFrame(preds).to_csv('houses.csv', index_label="Index", header=["Hogwarts House"])

def test_model(model, df):
    df = df[~df.isnull().any(axis=1)]
    x = df[COURSES].to_numpy()
    x = normalize(x)
    preds = perform_classification(model, x)
    preds = translate_pred(preds)
    df.insert(0, "Hogwarts House", preds)
    save_preds(preds)
    pass

def main():
    try:
        df, model = load_files()
    except Exception as e:
        print(e, file=sys.stderr)
        return
    test_model(model, df[COURSES])
    
    

if __name__ == "__main__":
    main()