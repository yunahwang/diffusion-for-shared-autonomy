import torch
import pandas as pd
import sys
from pathlib import Path
import os

def read_pkl(path):
    try:
        data = torch.load(path, map_location='cpu', weights_only=False)
        print(f"Loaded with torch.load: {type(data)}")
        print(len(data))
        print(data[0])
        return data
    except Exception as e:
        print(f"torch.load failed: {e}")
        raise

if __name__ == '__main__':
    path = Path(sys.argv[1])
    csv_dir = Path(__file__).parent / "2026_csv_backup"
    os.makedirs(csv_dir, exist_ok=True)

    if path.is_dir():
        pkls = list(path.glob("*.pkl"))
        print(f"Found {len(pkls)} pkl files")
        for pkl_path in pkls:
            print(f"Processing {pkl_path.name}")
            data = read_pkl(pkl_path)
            df = pd.DataFrame(data)
            csv_path = csv_dir / pkl_path.with_suffix(".csv").name
            df.to_csv(csv_path, index=False)
            print(f"Saved to {csv_path}")
    else:
        data = read_pkl(path)
        df = pd.DataFrame(data)
        csv_path = csv_dir / path.with_suffix(".csv").name
        df.to_csv(csv_path, index=False)
        print(f"Saved to {csv_path}")