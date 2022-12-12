import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

classes = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]

def load_dataset():
    df = pd.read_csv("datasets/dataset_train.csv", index_col="Index")
    return df

def show_histogram(df):
    Ravenclaw_df = df[df["Hogwarts House"] == "Ravenclaw"]
    Slytherin_df = df[df["Hogwarts House"] == "Slytherin"]
    Gryffindor_df = df[df["Hogwarts House"] == "Gryffindor"]
    Hufflepuff_df = df[df["Hogwarts House"] == "Hufflepuff"]
    df = df[classes]
    plt.title("Arithmancy")
    plt.hist(Ravenclaw_df["Arithmancy"].to_numpy(), bins=20, label="Ravenclaw", alpha=0.5)
    plt.hist(Slytherin_df["Arithmancy"].to_numpy(), bins=20, label="Slytherin", alpha=0.5)
    plt.hist(Gryffindor_df["Arithmancy"].to_numpy(), bins=20, label="Gryffindor", alpha=0.5)
    plt.hist(Hufflepuff_df["Arithmancy"].to_numpy(), bins=20, label="Hufflepuff", alpha=0.5) 
    plt.legend()
    plt.show()

def main():
    try:
        df = load_dataset()
    except Exception as e:
        print(e, file=sys.stderr)
        return
    show_histogram(df)
    
    

if __name__ == "__main__":
    main()