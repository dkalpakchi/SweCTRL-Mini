import json
import string
import re

import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


CONST = {
    'FormatErr': "Formatting errors",
    'SemJunk': 'Incomprehensible',
    'BadFlow': 'Incoherent',
    'Junk': "Junk text",
    'RepText': "Repetitive chunks"
}

OTHER_EXTRA = {
    'Konstigt att säga att sista dag att ansöka "var" ett visst datum -- låter som detta var i det förflutna.',
    'Formatting errors -- because of bug in the code',
    'Inkonsekvent stavning av "VD"',
    '"ta dig in i Europa med båt" låter mer som flyktingresor än som semesterresor',
    'The last bit of text is completely off-topic',
    'hoppas på att fånga?',
    '"o" instead of "och"',
    '"The Saints" is for Southampton FC',
    'Syntax errors',
    '"Riomaggiore o Monteross": "o" instead of "och"',
    '"vantar o strumpor funkar" - vardaglig, inte lämpligt för en tidningsartikel',
    '"fram o tillbaka": "o" instead of "och"',
    'Salvia',
    'Wrong English',
    '"vad vår mamma heter – för det har hon alltid gjort oavsett vart man kommer ifrån eller vem hennes pappa är egentligen" -- weird',
    'Beginning of forum',
    'The Magpies = Newcastle, not Manchester United',
    'På slutet glider ämnet iväg från en annons till något annat.'
}

OTHER_ERRORS = {
    'Punctuation error': CONST["FormatErr"],
    "Punctuation errors": CONST['FormatErr'],
    "Formatting errors.": CONST['FormatErr'],
    "Formatting errors": CONST['FormatErr'],
    'Skiljetecken ".," ser fel ut.': CONST["FormatErr"],
    'Det finns nästan inga skiljetecken.': CONST['FormatErr'],
    'Nästan inga skiljetecken.': CONST['FormatErr'],
    'Annonsen innehåller några helt galna meningar.': CONST["SemJunk"],
    'Skiljetecken saknas': CONST['FormatErr'],
    'Junk text': CONST['Junk'],
    'Annonsen innehåller en jättelång, helt galen mening.': CONST['SemJunk'],
    'Skiljetecken saknas. Kanske är detta en uppräkning.': CONST['FormatErr'],
    'Inga skiljetecken': CONST['FormatErr'],
    'En del skiljetecken saknas': CONST['FormatErr'],
    'Vissa skiljetecken saknas': CONST['FormatErr'],
    'Ett skiljetecken saknas': CONST['FormatErr'],
    '"...Europa där bolaget har verksamhet idag men flyttar in på nya fräscha lokaler om ca 2 år" är en knäpp mening': CONST['SemJunk'],
    'Helt osammanhängande': CONST["BadFlow"],
    'Börjar bra, men sedan är de flesta meningarna obegripliga': CONST['SemJunk'],
    'Semantiskt helt galen.': CONST["SemJunk"],
    'Texten är rörig, men grammatiskt korrekt.': CONST['SemJunk'],
    'Två identiska meningar efter varandra. Inte direkt ett grammatiskt fel, men det är väl ett fel.': CONST['RepText'],
    'Repetativ text': CONST['RepText'],
    'Repetativ text.': CONST['RepText'],
    'Rörig text utan fokus': CONST['SemJunk'],
    'Svårt att förstå vad författaren egentligen vill ha sagt.': CONST['SemJunk'],
    'Osammanhängande text': CONST['BadFlow'],
    'Osammanhängande': CONST['BadFlow'],
    'Repetativ text: "är känd för sitt kristallklara vatten och sin vackra natur. Ön är också känd för sitt kristallklara vatten och sin vackra natur"': CONST['RepText'],
    'Frasen "Foto : Shutterstock. com" upprepas hela tiden.': CONST['RepText'],
    'Repetitions': CONST['RepText'],
    'Flow errors': CONST['BadFlow'],
    'Repetitive text': CONST['RepText']
}


def minify(x):
    x = re.sub("[{}]".format(string.punctuation), "", x)
    x = re.sub("[0-9]", "", x)
    x = x.replace("\n", "").replace(" ", "")
    return x.strip()


if __name__ == '__main__':
    source_files = {}
    for x in ('p0', 'p1'):
        source_files[x] = {
            'data': ['swectrl_edata_{}.json'.format(x), 'swectrl_aedata_{}.json'.format(x)],
            'keys': ['swectrl_ekey_{}.json'.format(x), 'swectrl_aekey_{}.json'.format(x)]
        }

    eval_files = {
        'p0': 'res_p0.json',
        'p1': 'res_p1.json'
    }

    with open("prompts.yaml") as f:
        prompts = yaml.load(f, yaml.Loader)
        p2c = {
            p: "{}|{}".format(h, c)
            for h, v in prompts.items()
            for c, ps in v.items()
            for p in ps
        }
    
    df_data = []
    for p_id in ('p0', 'p1'):
        sf = source_files[p_id]
        ef = eval_files[p_id]
        
        orig_data = []
        for fn in sf['data']:
            with open(fn) as f:
                orig_data.extend(json.load(f))
        
        with open(ef) as f:
            res_data = json.load(f)['data']
   
        assert (len(orig_data) == len(res_data)), "[{}] Different sizes of original and annotated data!\n{} original vs {} annotated".format(
            p_id, len(orig_data), len(res_data)
        )

        for a, b in zip(orig_data, res_data):
            assert (minify(a['text'])[:70] == minify(b['context'])[:70]), "[{}] Found unequal datapoints:\n{}\n{}".format(p_id, a, b)
        print("[{}] ALL GOOD!".format(p_id))

        key = []
        for fn in sf['keys']:
            with open(fn) as f:
                key.extend(json.load(f))

        for r_data, key_data in zip(res_data, key):
            if isinstance(key_data, str):
                model = key_data.upper()
            elif isinstance(key_data, dict):
                model = 'SweCTRL-Mini ($r = {1}, p = {0}$)'.format(key_data['top_p'], key_data['repetition_penalty'])
            else:
                raise NotImplementedError("Not recognized format for a model")

            dp_id = "{}_{}".format(p_id, r_data['num'])

            prompt_type = None
            ctx = minify(r_data['context'])
            for p in p2c:
                if minify(p) in ctx:
                    prompt_type = p2c[p]
                    break

            assert prompt_type is not None, "Couldn't find a prompt\n{}".format(r_data['context'])

            cat, prompt_type = prompt_type.split("|")

            for ann in r_data['annotations']:
                errors, other_errors, fully_correct = set(), set(), False
                if ann.get('labels'):
                    for lab in ann['labels']:
                        errors.add(lab['marker']['name'])

                if ann.get('inputs'):
                    for inp in ann['inputs']:
                        if inp['marker']['name'] == 'Fully correct':
                            fully_correct = True
                        elif inp['marker']['name'] == 'Other errors':
                            notes = inp['content'].split("\n")
                            for note in notes:
                                note = note.strip()
                                if not note: continue
                                if note in OTHER_ERRORS:
                                    other_errors.add(OTHER_ERRORS[note])
                                elif note in OTHER_EXTRA:
                                    continue
                                else:
                                    raise ValueError("Not known error: {}".format(note))
                        else:
                            errors.add(inp['marker']['name'])

                if fully_correct:
                    assert len(errors) == 0, "Fully correct with errors?\n({}) {}".format(
                        dp_id, errors
                    )
                    df_data.append({
                        "model": model,
                        "datapoint": dp_id,
                        "error": None,
                        "is_main": True,
                        "category": cat.replace("_", "/"),
                        'Prompt type': prompt_type
                    })
                else:
                    if not errors and not other_errors:
                        print("Not correct, but no errors!\n", dp_id, r_data)
                        print()

                    for et, is_main in zip([errors, other_errors], [True, False]):
                        for e in list(et):
                            df_data.append({
                                "model": model,
                                "datapoint": dp_id,
                                "error": e,
                                "is_main": is_main,
                                "category": cat.replace("_", "/"),
                                'Prompt type': prompt_type
                            })
    
    df = pd.DataFrame.from_dict(df_data)
    df['Model'] = pd.Categorical(
        df['model'], sorted(df['model'].unique())
    )
    print(df.shape)
    print(df.head())

    # Seaborn settings
    sns.set(font_scale=1.2)
    sns.set_theme(style="dark")
    
    g = sns.displot(
        kind='hist',
        data=df[df['is_main']],
        x='category',
        hue='error',
        col='Model',
        discrete=True,
        multiple='stack'
    )
    for ax in g.axes[0]:
        ax.grid(axis='y')
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    plt.savefig("he_main_errors_by_cat.pdf", bbox_inches='tight')

    g = sns.displot(
        kind='hist',
        data=df[~df['is_main']],
        x='category',
        hue='error',
        col='Model',
        discrete=True,
        multiple='stack'
    )
    for ax in g.axes[0]:
        ax.grid(axis='y')
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    plt.savefig("he_other_errors_by_cat.pdf", bbox_inches='tight')

    g = sns.displot(
        kind='hist',
        data=df[df['is_main']],
        x='Prompt type',
        hue='error',
        col='Model',
        discrete=True,
        multiple='stack'
    )
    for ax in g.axes[0]:
        ax.grid(axis='y')
    plt.savefig("he_main_errors_by_prompt.pdf", bbox_inches='tight')
    
    g = sns.displot(
        kind='hist',
        data=df[~df['is_main']],
        x='Prompt type',
        hue='error',
        col='Model',
        discrete=True,
        multiple='stack'
    )
    for ax in g.axes[0]:
        ax.grid(axis='y')
    plt.savefig("he_other_errors_by_prompt.pdf", bbox_inches='tight')

    cols = ['model', 'category', 'Prompt type', 'datapoint']
    gdf = df[cols + ['error']].groupby(cols).count().reset_index()
    gdf = gdf.rename(columns={'error': "Error kinds"})
    gdf['Model'] = pd.Categorical(
        gdf['model'], sorted(gdf['model'].unique())
    )
    gdf['Has errors?'] = (gdf['Error kinds'] > 0).apply(lambda x: "Yes" if x else "No")
    print(gdf.shape)
    print(gdf.head())

    print(gdf.groupby(['Model', 'Error kinds']).count())
    
    g = sns.displot(
        kind='hist',
        data=gdf, 
        x='Has errors?',
        hue='Error kinds',
        col='Model',
        multiple='stack'
    )
    for ax in g.axes[0]:
        ax.grid(axis='y') 
    plt.savefig("he_has_errors.pdf", bbox_inches='tight')
    
    g = sns.displot(
        kind='hist',
        data=gdf, 
        x='category',
        hue='Error kinds',
        col='Model',
        multiple='stack'
    )
    for ax in g.axes[0]:
        ax.grid(axis='y') 
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    plt.savefig("he_num_errors_cat.pdf", bbox_inches='tight')
    
    g = sns.displot(
        kind='hist',
        data=gdf, 
        x='Prompt type',
        hue='Error kinds',
        col='Model',
        multiple='stack'
    )
    for ax in g.axes[0]:
        ax.grid(axis='y') 
    plt.savefig("he_num_errors_prompt.pdf", bbox_inches='tight')
