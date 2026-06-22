from sklearn.metrics import roc_curve, auc, roc_auc_score
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_info(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auroc = roc_auc_score(labels, scores)
    
    return tpr, fpr, auroc

def plot_auroc(tpr, fpr, score):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Random guess line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.savefig('roc_act_theta_id_vs_ood.png', dpi=150)

def read_csvs(id_path, ood_path):
    id_df = pd.read_csv(id_path)
    ood_df = pd.read_csv(ood_path)

    id_ecdf = id_df["ecdf_score"]
    ood_ecdf = ood_df["ecdf_score"]

    scores = np.concatenate([id_ecdf, ood_ecdf])
    labels = np.concatenate([np.zeros(len(id_ecdf)), np.ones(len(ood_ecdf))])

    return labels, scores


def main():
    # read csv name as argv
    # base = Path(__file__).parents[1]
    id_full_path = Path(sys.argv[1])
    ood_full_path = Path(sys.argv[2])

    # id_full_path = base / id_csv
    # ood_full_path = base / ood_csv

    # csv structure looks like
    # labels, scores (first, second col)
    labels, scores = read_csvs(id_full_path, ood_full_path)

    tpr, fpr, score = get_info(labels, scores)

    plot_auroc(tpr, fpr, score)



if __name__ == "__main__":
    main()