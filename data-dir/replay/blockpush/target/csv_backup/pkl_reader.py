import os
import torch
import pandas as pd

cols = [
    "st_block_x", "st_block_y", "st_block_or",
    "st_ee_x", "st_ee_y",
    "st_ee_tgt_x", "st_ee_tgt_y",
    "act_x", "act_y", "qval"
]

all_pkls = sorted(f for f in os.listdir(os.getcwd()) if f.endswith(".pkl"))
print("how many pkls,", len(all_pkls))

for pkl in all_pkls:
    pkl_full_path = os.path.join(os.getcwd(), pkl)
    pkl_pandas_name = os.path.splitext(pkl)[0]

    data = torch.load(pkl_full_path, map_location="cpu", weights_only=False)
    print(pkl, data.shape)

    df = pd.DataFrame(data, columns=cols)
    print(df.head())

    df.to_csv(pkl_pandas_name + ".csv", index=False)

print("done")