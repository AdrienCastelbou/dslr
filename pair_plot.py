import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns

classes = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]


def load_dataset():
    df = pd.read_csv("datasets/dataset_train.csv", index_col="Index")
    return df

def show_pair_plot(df):
    df = df[["Hogwarts House"] + classes]
    sns.pairplot(df,  diag_kind="hist", hue="Hogwarts House", markers='.', height=2)
    plt.show()


def main():
    try:
        df = load_dataset()
    except Exception as e:
        print(e, file=sys.stderr)
        return
    show_pair_plot(df)
    

if __name__ == "__main__":
    main()