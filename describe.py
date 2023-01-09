import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import sys
from TinyStatistician import TinyStatistician as TS

def load_dataset():
    if len(sys.argv) != 2:
        raise Exception("Error : Wrong number of arguments -> Usage : python3 describe.py path/to/dataset.csv")
    df = pd.read_csv(sys.argv[1], index_col="Index")
    return df

def describe_feature(datas):
    datas = datas[~np.isnan(datas)]
    infos = []
    infos.append(len(datas))
    infos.append(TS.mean(datas))
    infos.append(TS.std(datas))
    infos.append(np.min(datas))
    infos.append(TS.percentile(datas, 25))
    infos.append(TS.percentile(datas, 50))
    infos.append(TS.percentile(datas, 75))
    infos.append(np.max(datas))
    infos.append(TS.var(datas))
    return infos

def describe_df(df):
    described = pd.DataFrame(index=pd.Index(["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max", "Var"]))
    for feature in df:
        if not is_numeric_dtype(df[feature]) or df[feature].isnull().values.all():
            continue
        datas = df[feature].to_numpy()
        described[feature] = describe_feature(datas)
    print(described)

def main():
    try:
        df = load_dataset()
    except Exception as e:
        print(e, file=sys.stderr)
        return
    describe_df(df)
    

if __name__ == "__main__":
    main()