import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

classes = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]


def load_dataset():
    df = pd.read_csv("datasets/dataset_train.csv", index_col="Index")
    return df

def show_scatter_plot(df):
    Ravenclaw_df = df[df["Hogwarts House"] == "Ravenclaw"]
    Slytherin_df = df[df["Hogwarts House"] == "Slytherin"]
    Gryffindor_df = df[df["Hogwarts House"] == "Gryffindor"]
    Hufflepuff_df = df[df["Hogwarts House"] == "Hufflepuff"]
    #df.drop('Hogwarts House', inplace=True, axis=1)
    df = df[classes]
    for x_feature in df:
        for y_feature in df:
            if x_feature != y_feature:
                plt.scatter(Ravenclaw_df[x_feature].to_numpy(), Ravenclaw_df[y_feature].to_numpy(), label="Ravenclaw", alpha=0.5)
                plt.scatter(Slytherin_df[x_feature].to_numpy(), Slytherin_df[y_feature].to_numpy(), label="Slytherin", alpha=0.5)
                plt.scatter(Gryffindor_df[x_feature].to_numpy(), Gryffindor_df[y_feature].to_numpy(), label="Gryffindor", alpha=0.5)
                plt.scatter(Hufflepuff_df[x_feature].to_numpy(), Hufflepuff_df[y_feature].to_numpy(), label="Hufflepuff", alpha=0.5)
                plt.xlabel(x_feature)
                plt.ylabel(y_feature)
                plt.legend()
                plt.show()

def main():
    try:
        df = load_dataset()
    except Exception as e:
        print(e, file=sys.stderr)
        return
    show_scatter_plot(df)
    

if __name__ == "__main__":
    main()