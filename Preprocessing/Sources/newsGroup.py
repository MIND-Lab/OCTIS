import pandas as pd

def retrieve():
    df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
    diz = df.to_dict()
    labelstmp = list(diz['target_names'].values())
    labels = []
    for label in labelstmp:
        labels.append([label])
    corpus = list(diz['content'].values())
    result = {}
    result["corpus"] = corpus
    result["docLabels"] = labels
    return result
