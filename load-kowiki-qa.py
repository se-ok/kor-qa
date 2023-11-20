import json

from tqdm import tqdm

FILE = 'generated_qa.jsonl'

qa_dicts = []

with open(FILE) as f:
    json_strings = []

    pbar = tqdm(f)
    for line in pbar:
        json_strings.append(line)
        if line == '}\n':
            json_str = '\n'.join(json_strings)
            obj = json.loads(json_str)
            qa_dicts.append(obj)

            pbar.set_postfix_str(f'{len(qa_dicts)} Q&A loaded')
            
            json_strings = []
