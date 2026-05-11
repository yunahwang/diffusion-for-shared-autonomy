import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def main(f_path):
    df = pd.read_csv(f_path)
    loss = df["loss"].values
    # print(loss)

    plt.plot(loss)
    plt.show()


if __name__ == "__main__":
    f_path = Path(sys.argv[1])
    main(f_path)


    # reads loss dataframe file from wandb:
    """
    import wandb
    api = wandb.Api()
    run = api.run("/yhwang56-university-of-wisconsin-madison/diffusha/runs/tr3wtwfz")
    print(run.history()) <- this gives a dataframe object
    """
