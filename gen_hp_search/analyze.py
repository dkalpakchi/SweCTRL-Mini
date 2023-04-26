import os
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from nltk.metrics.confusionmatrix import ConfusionMatrix

from control_codes import *


ECC = {
    None: {
        'color': 'grey',
        'label': 'No ECC',
    },
    False: {
        'color': 'red',
        'label': 'Wrong ECC'
    },
    True: {
        'color': 'green',
        'label': "Correct ECC"
    }
}


def ngrams(tokens, n=3, stride=1):
    return zip(*[tokens[i::stride] for i in range(0, n)])


def expand_cat(x):
    if x == "lit":
        return "literature"
    else:
        return x


def check_repeated_words(text, max_window=5):
    repeats = defaultdict(int)

    # very rough version of tokens
    tokens = [x.strip() for x in text.split()]
    num_tokens = len(tokens)
    
    for w in range(1, max_window+1):
        for i in range(w, num_tokens):
            if tokens[i:i+w] == tokens[i-w:i]:
                repeats[w] += 1
    return repeats


def analyze_text(text, cc):
    end_cc = END_C_CODES[cc]

    stats = {
        'actual_ecc': None,
        'correct_ecc': cc,
        'ecc': None
    }
    correct_ecc = False
    for cat, code in END_C_CODES.items():
        if code == end_cc and code in text:
            correct_ecc = True
        elif code in text:
            stats['ecc'] = False
            stats['actual_ecc'] = cat
    
    if stats['ecc'] is None and correct_ecc:
        stats['ecc'] = True
        stats['actual_ecc'] = cc
    
    rough_tokens = text.split()
    repeats = check_repeated_words(text).keys()
    stats[var["max_repeats"]] = max(repeats) if repeats else 0
    return stats


# Adapted from:
# https://stackoverflow.com/questions/28546253/how-to-create-request-body-for-python-elasticsearch-msearch
def msearch(client, search_arr):
    request = ''
    for each in search_arr:
        request += '{} \n'.format(json.dumps(each))
    return client.msearch(body = request)


def compute_overlap_es(client, text, ng=13):
    def get_query(msg, for_msearch_api=False):
        q =  {
            "query": {
                "match_phrase": {
                    "text": msg
                }
            }
        }
        if for_msearch_api:
            q['size'] = 0
            q['track_total_hits'] = True
        return q

    # print(client.count(index='mc4_ngrams_13_1', body=query))
    ng2index = {
        13: ['mc4_ngrams_13_1', 'runeberg_ngrams_13_1'],
        7: ['mc4_ngrams_7_1', 'runeberg_ngrams_7_1']
    }
    index_names = ng2index.get(ng)
    total_ng, found_ng = 0, 0
    search_arr, res = {}, {}
    for index_name in index_names:
        search_arr[index_name] = []
        res[index_name] = []
    for ngram in ngrams(text.split(), n=ng):
        total_ng += 1
        for index_name in index_names:
            str_ngram = " ".join(ngram)
            # req_head
            search_arr[index_name].append({'index': index_name})
            # req_body
            search_arr[index_name].append(get_query(str_ngram, for_msearch_api=True))
   
    for index_name in index_names:
        if search_arr[index_name]:
            res[index_name] = msearch(client, search_arr[index_name])
  
    if sum([len(x) for x in res.values()]) == 0:
        return 0

    for resp in zip(*[x['responses'] for x in res.values() if x]):
        final_count = sum([r['hits']['total']['value'] for r in resp if r])
        found_ng += final_count > 0
    return found_ng / (total_ng or 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, required=True, help="Path to the output folder of generate script")
    parser.add_argument('-o', '--output', type=str, default="", help="Path to save the images")
    parser.add_argument('-r', '--rep_p', default=-1, type=float)
    parser.add_argument('-p', '--top_p', default=-1, type=float)
    parser.add_argument('-t', '--temp', default=-1, type=float)
    parser.add_argument('-es', '--elasticsearch', action='store_true', help="Whether ElasticSearch is available for calculations")
    args = parser.parse_args()

    font = {
        'size'   : 18
    }
    mpl.rc('font', **font)

    var = {
        'rep_p': "$r$",
        "num_tokens": "Number of tokens",
        "ecc": "ECC present?",
        'max_repeats': "Loop size"
    }

    if args.elasticsearch:
        from dotenv import load_dotenv
        from elasticsearch import Elasticsearch
        
        load_dotenv()
        ELASTIC_PASSWORD = os.getenv('ES_PWD')
        client = Elasticsearch(
            "https://localhost:9200",
            ca_certs=os.path.join("gen_hp_search", "ca.crt"),
            basic_auth=("elastic", ELASTIC_PASSWORD),
            request_timeout=5000
        )

    # Thing to test MultiSearch API to track exact counts
    # search_arr = []
    # search_arr.append({'index': 'mc4_ngrams_13_1'})
    # search_arr.append(get_query("Riksdagen tillkännager för regeringen som sin mening", for_msearch_api=True))
    # search_arr.append({'index': 'mc4_ngrams_13_1'})
    # search_arr.append(get_query("tillkännager för regeringen", for_msearch_api=True))
    # res = msearch(client, search_arr)
    # print(res)

    # print(client.count(index='mc4_ngrams_13_1', body=get_query("Riksdagen tillkännager för regeringen som sin mening")))
    # print(client.count(index='mc4_ngrams_13_1', body=get_query("tillkännager för regeringen")))
    # sss.exit()

    if args.output and not os.path.exists(args.output):
        os.makedirs(args.output)

    texts = {
        'temp': defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        'top_p': defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    }
    data = []
    for d in os.listdir(args.folder):
        cat_dir = os.path.join(args.folder, d)
        if os.path.isdir(cat_dir):
            print("Working with {} category".format(d))
            for fname in os.listdir(cat_dir):
                with open(os.path.join(cat_dir, fname)) as f:
                    res = json.load(f)
                    cc = res['control']
                    rep_p = round(res['repetition_penalty'], 2)
                    t = res.get('temperature')
                    top_p = res.get('top_p')

                    data.append({
                        'code': cc,
                        var['rep_p']: rep_p,
                        'temp': round(t, 2) if t else t,
                        'top_p': round(top_p, 2) if top_p else top_p,
                        var['num_tokens']: res['total_generated'],
                    })

                    text = res['text']
                    if t:
                        texts['temp'][cc][rep_p][round(t, 2)].append(text)
                    elif top_p:
                        texts['top_p'][cc][rep_p][round(top_p, 2)].append(text)

                    text_stats = analyze_text(text, cc)
                    data[-1].update(text_stats)


    df = pd.DataFrame.from_dict(data)

    codes = df['code'].unique()
    df[var["ecc"]] = df["ecc"].apply(lambda x: ECC[x]['label'])

    hyp_p = {
        'top_p': "$p$",
        'temp': "$T$"
    }

    ranges = {
        "top_p": [round(x, 2) for x in np.arange(0.7, 1.02, 0.05)],
        "temp": [round(x, 2) for x in np.arange(0, 1.1, 0.2)],
    }

    ecc_keys = ECC.keys()
    ecc = {
        'colors': [ECC[x]['color'] for x in ecc_keys],
        'labels': [ECC[x]['label'] for x in ecc_keys]
    }

    if args.top_p >= 0 or args.rep_p >= 0 or args.temp >= 0:
        df['Category'] = df['code'].apply(expand_cat)
        df['Category'] = pd.Categorical(
            df['Category'], categories=sorted(df['Category'].unique())
        )
        fhp = {}
        for hp in hyp_p:
            hp_val = getattr(args, hp)
            if hp_val >= 0:
                df = df[df[hp] == hp_val]
                fhp[hp] = hp_val
     
        if args.rep_p >= 0:
            df = df[df[var["rep_p"]] == args.rep_p]
            fhp["rep_p"] = args.rep_p
        
        fhp_str = "-".join(["{}__{}".format(x, str(y).replace(".", "_")) for x, y in fhp.items()])

        with pd.option_context('mode.chained_assignment', None):
            for hp in fhp:
                if hp == "rep_p": continue
                df[hyp_p[hp]] = pd.Categorical(
                    df[hp].apply(str),
                    categories=map(str, ranges[hp])
                )

        cm = ConfusionMatrix(
            list(map(str, df["correct_ecc"].tolist())),
            list(map(str, df["actual_ecc"].tolist()))
        )
        g = sns.heatmap(
            cm._confusion,
            annot=True,
            cmap=sns.cm.rocket_r,
            xticklabels=cm._values,
            yticklabels=cm._values,
            annot_kws={"fontsize": 5},
            linewidths=0.1
        )

        for text in g.texts:
            if text.get_text() == '0': # conditional for cells one would like to highlight. This can then be changed to the highest value...
                text.set_text("")
        g.set_ylabel("Correct ECC", fontsize=8)
        g.set_xlabel("Actual ECC", fontsize=8)
        g.set_xticklabels(g.get_xticklabels(), rotation=90, ha='center', va='bottom', fontsize=7)
        g.set_yticklabels(g.get_yticklabels(), fontsize=7)
        cbar = g.collections[0].colorbar
        cbar.ax.tick_params(labelsize=8)
        g.xaxis.tick_top()
        if args.output:
            plt.savefig(
                os.path.join(args.output, "{}_ecc_cm.pdf".format(fhp_str)),
                bbox_inches='tight'
            )
        else:
            plt.show()

        g = sns.displot(
            data=df,
            x='Category', hue=var["ecc"],
            kind="hist",
            palette=ecc['colors'],
            hue_order=ecc['labels'],
            multiple="stack", discrete=True,
            height=4, aspect=4
        )
        g.add_legend()
        plt.subplots_adjust(right=0.85)
        for ax in g.axes[0]:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
        if args.output:
            plt.savefig(
                os.path.join(args.output, "{}_ecc.pdf".format(fhp_str)),
                bbox_inches='tight'
            )
        else:
            plt.show()

        g = sns.displot(
            data=df,
            x='Category', hue=var["max_repeats"],
            kind="hist",
            multiple="stack",
            height=4, aspect=4,
            palette=sns.light_palette((20, 60, 50), input="husl")
        )
        g.add_legend()
        plt.subplots_adjust(right=0.9)
        for ax in g.axes[0]:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11) 
        if args.output:
            plt.savefig(
                os.path.join(args.output, "{}_max_repeats.pdf".format(fhp_str)),
                bbox_inches='tight'
            )
        else:
            plt.show()
        
        g = sns.catplot(
            data=df, x='Category', y=var["num_tokens"],
            color='skyblue', kind='box',
            medianprops={"color": "coral"},
            height=4, aspect=4
        )
        for ax in g.axes[0]:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
        if args.output:
            plt.savefig(
                os.path.join(args.output, "{}_num_tokens.pdf".format(fhp_str)),
                bbox_inches='tight'
            )
        else:
            plt.show()

        text_sim_data = []
        rep_p = fhp.get("rep_p")
        overlap_key = "$O_{1}^{13}$"
        for hp in fhp:
            if hp == "rep_p": continue
            for code, dct in tqdm(texts[hp].items()):
                if rep_p is None:
                    for rep_p, rep_p_dct in dct.items():
                        for hp_val, text_lst in rep_p_dct.items():
                            if hp_val == fhp[hp]:
                                for text in text_lst:
                                    ref_texts = [t.split() for t in text_lst if t != text]
                                    if len(ref_texts) == 0:
                                        ref_texts = [text]
                                    text_sim_data.append({
                                        'code': code,
                                        var["rep_p"]: rep_p,
                                        hp: hp_val,
                                        "BLEU-4": sentence_bleu(
                                            ref_texts,
                                            text.split()
                                        )
                                    })
                                    if args.elasticsearch:
                                        text_sim_data[-1][overlap_key] = compute_overlap_es(
                                            client, text, ng=13
                                        )

                else:
                    for hp_val, text_lst in tqdm(dct[rep_p].items()):
                        if hp_val == fhp[hp]:
                            for text in text_lst:
                                ref_texts = [t.split() for t in text_lst if t != text]
                                if len(ref_texts) == 0:
                                    ref_texts = [text]
                                text_sim_data.append({
                                    'code': code,
                                    var["rep_p"]: rep_p,
                                    hp: hp_val,
                                    "BLEU-4": sentence_bleu(
                                        ref_texts,
                                        text.split()
                                    )
                                })
                                if args.elasticsearch:
                                    text_sim_data[-1][overlap_key] = compute_overlap_es(
                                        client, text, ng=13
                                    )

        text_sim = pd.DataFrame.from_records(text_sim_data)
        text_sim['Category'] = text_sim['code'].apply(expand_cat)
        text_sim['Category'] = pd.Categorical(
            text_sim['Category'], categories=sorted(text_sim['Category'].unique())
        )
        if len(text_sim[text_sim["BLEU-4"] > 0]) > 0:
            with pd.option_context('mode.chained_assignment', None):
                for hp in fhp:
                    if hp == "rep_p": continue
                    text_sim[hyp_p[hp]] = pd.Categorical(
                        text_sim[hp].apply(str),
                        categories=map(str, ranges[hp])
                    )
            g = sns.catplot(
                data=text_sim, x='Category', y="BLEU-4",
                color="skyblue", kind="box",
                medianprops={"color": "coral"},
                height=4, aspect=4
            )
            for ax in g.axes[0]:
                ax.set_ylim((0, 1))
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=11)
            if args.output:
                plt.savefig(
                    os.path.join(args.output, "{}_bleu.pdf".format(fhp_str)),
                    bbox_inches='tight'
                )
            else:
                plt.show()
        
        if len(text_sim[text_sim[overlap_key] > 0]) > 0:
            g = sns.catplot(
                data=text_sim, x='Category', y=overlap_key,
                color="skyblue", kind="box",
                medianprops={"color": "coral"},
                height=4, aspect=4
            )
            for ax in g.axes[0]:
                ax.set_ylim((0, 1))
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=11)
            if args.output:
                plt.savefig(
                    os.path.join(args.output, "{}_overlap.pdf".format(fhp_str)),
                    bbox_inches='tight'
                )
            else:
                plt.show()

    else:
        for code in codes:
            code_stats = df[df['code'] == code]
            #code_group = code_stats.groupby(by=["rep_p", "top_p"], group_keys=False).sum().reset_index()

            for hp in hyp_p:
                code_hp_stats = code_stats[code_stats[hp].notna()]

                with pd.option_context('mode.chained_assignment', None):
                    code_hp_stats[hyp_p[hp]] = pd.Categorical(
                        code_hp_stats[hp].apply(str),
                        categories=map(str, ranges[hp])
                    )
                g = sns.displot(
                    data=code_hp_stats,
                    x=hyp_p[hp], hue=var["ecc"],
                    kind="hist", col=var["rep_p"],
                    palette=ecc['colors'],
                    hue_order=ecc['labels'],
                    multiple="stack", discrete=True
                )
                g.add_legend()
                plt.subplots_adjust(right=0.9)
                if args.output:
                    plt.savefig(
                        os.path.join(args.output, "{}_{}_ecc.pdf".format(code, hp)),
                        bbox_inches='tight'
                    )
                else:
                    g.fig.suptitle(code)
                    plt.show()
                
                g = sns.displot(
                    data=code_hp_stats,
                    x=hyp_p[hp], hue=var["max_repeats"],
                    kind="hist", col=var["rep_p"],
                    multiple="stack", discrete=True,
                    palette=sns.light_palette((20, 60, 50), input="husl")
                )
                
                g.add_legend()
                plt.subplots_adjust(right=0.9)
                if args.output:
                    plt.savefig(
                        os.path.join(args.output, "{}_{}_max_repeats.pdf".format(code, hp)),
                        bbox_inches='tight'
                    )
                else:
                    g.fig.suptitle(code)
                    plt.show()
                
                sns.catplot(
                    data=code_hp_stats, x=hyp_p[hp], y=var["num_tokens"],
                    col=var["rep_p"], color='skyblue', kind='box',
                    medianprops={"color": "coral"}
                )
                if args.output:
                    plt.savefig(
                        os.path.join(args.output, "{}_{}_num_tokens.pdf".format(code, hp)),
                        bbox_inches='tight'
                    )
                else:
                    g.fig.suptitle(code)
                    plt.show()


                text_sim_data = []
                for rep_p in texts[hp][code]:
                    for hp_val, text_lst in texts[hp][code][rep_p].items():
                        for text in text_lst:
                            ref_texts = [t.split() for t in text_lst if t != text]
                            if len(ref_texts) == 0:
                                ref_texts = [text]
                            text_sim_data.append({
                                var["rep_p"]: rep_p,
                                hp: hp_val,
                                "BLEU-4": sentence_bleu(
                                    ref_texts,
                                    text.split()
                                )
                            })

                text_sim = pd.DataFrame.from_records(text_sim_data)
                if len(text_sim[text_sim["BLEU-4"] > 0]) > 0:
                    with pd.option_context('mode.chained_assignment', None):
                        text_sim[hyp_p[hp]] = pd.Categorical(
                            text_sim[hp].apply(str),
                            categories=map(str, ranges[hp])
                        )
                    g = sns.catplot(
                        data=text_sim, x=hyp_p[hp], y="BLEU-4",
                        col=var["rep_p"], color="skyblue", kind="box",
                        medianprops={"color": "coral"}
                    )
                    for ax in g.axes[0]:
                        ax.set_ylim((0, 1))
                    if args.output:
                        plt.savefig(
                            os.path.join(args.output, "{}_{}_bleu.pdf".format(code, hp)),
                            bbox_inches='tight'
                        )
                    else:
                        g.fig.suptitle(code)
                        plt.show()

