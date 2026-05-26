"""
Read target/randp_0.0's pickle files and perform data-augmentation
action y - action increments half the size, 1/3, 1/4, 1/5, 1/6, 1/7 the size (so episodes are actually longer)
assert needed, new folder creations needed
"""
import pickle
import torch
from pathlib import Path

def main(root, orig):
    # read pickle files in orig
    # and in mkdir folder names "2nd", "3rd", ... "7th" inside root
    # save pickles inside "2nd" > "pkl", ... and save csvs inside "2nd" > "csv", ....
    # discern episode start-end using the following code snippet "reconstruct_eps.py"
    # assert that every episode is lengthed by 2x, 3x, ..., 7x by dividing 
    # action_y column value into 1/2, 1/3, ..., 1/7
    # include assertion values on episode length and total row numbers post generation
    
    pass

if __name__ == "__main__":
    root = Path(__file__).parents[2] / "data-dir" / "replay" / "blockpush" / "target" 
    orig = root / "randp_0.0"
    main(root, orig)