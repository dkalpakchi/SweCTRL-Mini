import json
import numbers
import argparse
from collections import defaultdict
from pathlib import Path, PurePath

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from nltk.metrics.agreement import AnnotationTask
from sklearn.metrics import matthews_corrcoef, accuracy_score
from scipy.stats import spearmanr, describe

from rouge import Rouge


def normalize_float(x, code_to_remove):
    res_str = x.replace(code_to_remove, "").replace(" ", "").strip()
    try:
        return float(res_str)
    except:
        return np.nan

def normalize_yes_no(x, code_to_remove):
    codes = {
        "ja": 1,
        "kanske": 0,
        "nej": -1
    }
    res_str = x.replace(code_to_remove, "").lower().strip()
    return codes.get(res_str, -100)


def normalize_absaimm(x):
    return normalize_float(x, ":absaimm:$")


def normalize_sweparaphrase(x):
    return normalize_float(x, ":sweparaphrase:$")


def normalize_dalajged(x):
    return normalize_yes_no(x, ":dalajged:$")


def normalize_swefaq(x):
    return normalize_yes_no(x, ":swefaq:$")


def normalize_swenli(x):
    return normalize_yes_no(x, ":swenli:$")


def normalize_swewinograd(x):
    return normalize_yes_no(x, ":swewinograd:$")


def calc_rouge(y_true, y_pred):
    r = Rouge()
    y_true = {i: [v.strip()] for i, v in enumerate(y_true)}
    y_pred = {i: [v.strip()] for i, v in enumerate(y_pred)}
    return r.compute_score(y_true, y_pred)


def swefaq_pseudoalpha(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    num = 109 / 2049
    den = 1940 / 2049
    return (acc - num) / den


def krippendorf_alpha(y_true, y_pred):
    "3-tuples (coder,item,label)"
    data = []
    missing = 0
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        if isinstance(yp, numbers.Number) and not np.isnan(yp):
            data.append((0, i, yt))
            data.append((1, i, yp))
        else:
            missing += 1
    if data:
        at = AnnotationTask(data)
        res = at.alpha()
    else:
        res = None
    return {
        'alpha': res,
        'missing': missing / len(y_true)
    }

def spearman_rho(y_true, y_pred):
    y_filtered_true, y_filtered_pred = [], []
    missing = 0
    for i in range(len(y_pred)):
        x = y_pred[i]
        if isinstance(x, numbers.Number) and not np.isnan(x):
            y_filtered_pred.append(x)
            y_filtered_true.append(y_true[i])
        else:
            missing += 1
    return {
        'rho': spearmanr(y_filtered_true, y_filtered_pred),
        'missing': missing / len(y_true)
    }


metrics = {
    'absaimm': {
        "Krippendorf's alpha": krippendorf_alpha,
        "Spearman's correlation": spearman_rho
    },
    'swedn': {
        "ROUGE-L": calc_rouge,
    },
    'swewinograd': {
        "Krippendorf's alpha": krippendorf_alpha,
        "Accuracy": accuracy_score
    },
    'swefracas': {
        "Matthews correlation": matthews_corrcoef
    },
    'dalajged': {
        "Krippendorf's alpha": krippendorf_alpha,
        "Accuracy": accuracy_score
    },
    'swenli': {
        "Krippendorf's alpha": krippendorf_alpha,
        "Accuracy": accuracy_score
    },
    'swefaq': {
        "Krippendorf's pseudoalpha": swefaq_pseudoalpha,
        "Accuracy": accuracy_score
    },
    'sweparaphrase': {
        "Krippendorf's alpha": krippendorf_alpha
    },
    'swewic': {
        "Accuracy": accuracy_score 
    },
    'swewinogender': {
        "Krippendorf's alpha": krippendorf_alpha,
    },
    'swediagnostics': {
        "Krippendorf's alpha": krippendorf_alpha,
        "Matthews correlation": matthews_corrcoef
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, required=True, help="The folder with the experimental data")
    args = parser.parse_args()

    # args.folder should contain all experiments
    # each experiment is a separate folder containing many other folders, each with a specific fold

    stats = defaultdict(dict)
    for task_path in sorted(Path(args.folder).glob(str(PurePath("*", "eval_detailed.json")))):
        task_name = task_path.parent.name
        _, _, sl_task, n_epochs, _ = task_name.split("_")
        norm_fn = locals().get("normalize_{}".format(sl_task), lambda x: x)
        with open(task_path) as f:
            eval_res = json.load(f)
            task_stats = {
                'true': [],
                'pred': []
            }
            
            for fold_stats in eval_res:
                for x in fold_stats['data']:
                    pred = norm_fn(x['pred_str'])
                    label = norm_fn(x['label_str'])
                    task_stats['true'].append(label)
                    task_stats['pred'].append(pred)
        
        # Distribution of classes in the test data
        # Just to see if we could beat a theoretically best majority baseline
        class_dist = fold_stats['class_dist']
            

        if sl_task == 'swedn':
            summaries = []
            with open("swedn_baseline_summaries.txt") as f:
                for line in f:
                    summaries.append(line.strip())
            task_stats['base'] = summaries
        else:
            base_pred = norm_fn(max(class_dist.items(), key=lambda x: x[1])[0])
            task_stats['base'] = [base_pred for _ in range(len(task_stats['true']))]
        
        for m, m_fn in metrics[sl_task].items():
            if m_fn is not None:
                task_stats[m] = m_fn(task_stats['true'], task_stats['pred'])
                task_stats["{}_base".format(m)] = m_fn(task_stats['true'], task_stats['base'])

        del task_stats['true']
        del task_stats['pred']
        if 'base' in task_stats:
            del task_stats['base']
        stats[sl_task]['{} epochs'.format(n_epochs)] = task_stats

    for task in stats:
        print(task)
        print(stats[task])
        if task == 'swedn':
            data = []
            for k, v in stats[task].items():
                for val in v['ROUGE-L'][1]:
                    data.append(
                        {'epochs': k, 'ROUGE-L': val}
                    )
                print(k)
                print("\t", describe(v['ROUGE-L'][1]))
                print("\tQ1, Q2, Q3: ", np.quantile(
                    v['ROUGE-L'][1], [0.25, 0.5, 0.75]
                ))
            else:
                for val in v['ROUGE-L_base'][1]:
                    data.append(
                        {'epochs': 'base', 'ROUGE-L': val}
                    )

            df = pd.DataFrame.from_dict(data)
            df['Model'] = pd.Categorical(
                df['epochs'],
                categories=['base', '1 epochs', '3 epochs', '5 epochs']
            )
            print(df.head())
            
            g = sns.catplot(
                data=df, x='Model', y='ROUGE-L',
                color='skyblue', kind='box',
                medianprops={"color": "coral"}
            )
            g.set(xlabel=None)
            plt.savefig("swedn_rouge.pdf", bbox_inches='tight')
        print()

