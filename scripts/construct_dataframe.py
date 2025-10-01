import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm


CONFIG_PATTERN = r'^(?P<model>.+?)-ctx(?P<context>\d+)-(?P<interpolation>linear|cubic|quadratic)$'
METRIC_COL = 'ndcg_at_10'

dataset_aliases = {
    'LEMBNarrativeQARetrieval': 'NQA',
    'LEMBSummScreenFDRetrieval': 'SFD',
    'LEMBQMSumRetrieval': 'QMS',
    'LEMBWikimQARetrieval': 'WQA',    
}

# gather results
df = pd.DataFrame()
paths = [p for p in Path('results').rglob('*jsonl') if p.stem != 'model_meta']
for path in tqdm(paths, desc='Gathering results'):
    # load file
    _df = pd.read_json(path, lines=True)
    _df.drop('task', axis=1, inplace=True)
    _df[METRIC_COL] *= 100
    
    # infer config from path
    config = re.match(CONFIG_PATTERN, path.parent.name).groupdict()
    for param, value in config.items():
        if param == 'model':
            value = value.split('/')[-1]
        _df[param] = value

    # add dataset alias
    _df['dataset'] = dataset_aliases[path.stem]
    
    df = pd.concat([df, _df])

# save dataframe
dataframe_path = Path(f'data/{METRIC_COL}.csv')
df.to_csv(dataframe_path, index=False)
print(f'Dataframe was saved to "{dataframe_path}"')
