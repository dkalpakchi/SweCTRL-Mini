import os
import glob
import csv
import argparse
from operator import itemgetter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', type=str, required=True, help="CSV-separated files with the perplexities for each checkpoint")
    args = parser.parse_args()

    ITERS_IN_EPOCH = 97784

    df = None
    for pat in args.files.split(","):
        for fname in glob.glob(pat):
            ds = os.path.basename(fname).replace("ppl_", "").replace(".csv", "").strip()
            with open(fname.strip()) as f:
                reader = csv.reader(f)
                rows = []
                for row in reader:
                    checkpoint, ppl = row
                    rows.append([
                        int(checkpoint.split("-")[-1]) / ITERS_IN_EPOCH,
                        round(float(ppl), 2)
                    ])

            rows = sorted(rows, key=itemgetter(0))
            c_df = pd.DataFrame.from_records(
                rows, columns = ["epoch", ds]
            )
            if df is None:
                df = c_df
            else:
                df = df.join(c_df.set_index("epoch"), on='epoch')

    df = df.set_index('epoch')
    print(df.head())
    sns.lineplot(df)
    plt.xlim(0)
    plt.ylim(0)
    plt.ylabel("perplexity")
    plt.savefig("perplexity.pdf", bbox_inches='tight')
    plt.show()
