'''Generated Korean Q&A from Wikipedia.

Parsed plain text files are from wikiextractor.
'''
from concurrent.futures import ThreadPoolExecutor
import json
from pprint import pprint
import random
import re
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import openai
from openai.types.chat.chat_completion import ChatCompletion

TEXT_ROOT = 'kowiki-20231101'
REMOVE_PATTERN = r"&lt;.*?&gt;"
NUM_DOCS_PER_SIZE = 1000
NUM_THREADS = 5
OUTPUT_FILE = 'generated_qa.jsonl'

def plain_texts(file: str) -> list[str]:
    with open(file) as f:
        docs = []
        lines = []

        for line in f:
            line = line.strip()
            if line.startswith('<doc '):
                continue

            line = re.sub(REMOVE_PATTERN, '', line).strip()

            if line == '</doc>':
                doc = '\n'.join(lines)
                while '\n\n\n' in doc:
                    doc = doc.replace('\n\n\n', '\n\n')

                docs.append(doc)
                lines = []

            else:
                lines.append(line)

    return docs

def text_files() -> list[Path]:
    filenames = []

    root = Path(TEXT_ROOT)
    for cat in root.iterdir():
        filenames.extend(cat.iterdir())

    return filenames

def get_kowiki_as_string() -> list[str]:
    docs = []

    for f in tqdm(text_files()):
        docs.extend(plain_texts(f))

    return docs

def generate(context: str, model='gpt-3.5-turbo-1106', timeout=10, **kwargs) -> dict:
    '''Given a context, returns the following:
    {
        'model': str,
        'context': str,
        'generated': str,
        'question': str | None,
        'answer': str | None,
        'input_tokens': int,
        'output_tokens': int,
    }

    If `generated` cannot be parsed, `question` and `answer` will be set to None.
    '''
    if 'n' in kwargs and kwargs['n'] != 1:
        raise ValueError("")

    client = openai.OpenAI()

    message = f'다음 지문을 써야 제대로 답할 수 있는 질문을 딱 하나 만들고, 그에 대한 대답을 제시해 주세요.\n지문:\n{context}'

    try:
        completion: ChatCompletion = client.chat.completions.create(messages=[
            {'role': 'system', 'content': '지시에 따라 질문과 대답을 만들어주세요.'},
            {'role': 'user', 'content': message}
        ], model=model, timeout=timeout, **kwargs)
    except openai.APITimeoutError:
        return {
            'model': model,
            'context': context,
            'generated': None,
            'question': None,
            'answer': None,
            'input_tokens': 0,
            'output_tokens': 0,
        }

    generated = completion.choices[0].message.content
    split = generated.split('대답:')

    if generated.startswith('질문:') and len(split) == 2:
        question, answer = split
        question = question[3:].strip()
        answer = answer.strip()

    else:
        question, answer = None, None

    return {
        'model': model,
        'context': context,
        'generated': generated,
        'question': question,
        'answer': answer,
        'input_tokens': completion.usage.prompt_tokens,
        'output_tokens': completion.usage.completion_tokens,
    }


def main():
    docs = get_kowiki_as_string()

    docs_by_size: dict[str, list] = defaultdict(list)

    for doc in docs:
        docs_by_size[len(doc) // 100].append(doc)

    chosen_docs = []
    for hundred in range(5, 26):
        copied_docs = docs_by_size[hundred][:]
        random.shuffle(copied_docs)

        chosen_docs.extend(copied_docs[:NUM_DOCS_PER_SIZE])

    average_length = sum(len(doc) for doc in chosen_docs) / len(chosen_docs)
    print(f'{len(chosen_docs)} docs of average length {int(average_length)} have been selected.')

    input_tokens = 0
    output_tokens = 0
    success = 0
    failed = 0
    timeout = 0

    pbar = tqdm(desc='Generating')

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
        for result in pool.map(generate, chosen_docs):
            pprint(result)

            input_tokens += result['input_tokens']
            output_tokens += result['output_tokens']

            if result['generated'] is None:
                timeout += 1
            elif result['question'] is None:
                failed += 1
            else:
                success += 1

            pbar.set_postfix_str(f'Success:{success},Failed:{failed},Timeout:{timeout},In:{input_tokens},Out:{output_tokens}')
            pbar.update()
            print('')

            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                dump = json.dumps(result, indent=2, ensure_ascii=False)
                print(dump, file=f)


if __name__ == '__main__':
    main()
