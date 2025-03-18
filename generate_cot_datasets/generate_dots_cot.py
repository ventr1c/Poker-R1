import json
import re
import argparse

from verl.utils.reward_score.countdown import compute_score
from transformers import AutoTokenizer

def replace_think_tags(text, replace_with=' .', num_repeats=None):
    if num_repeats is None:
        pattern = re.compile(r'(<think>)(.*?)(</think>)', re.DOTALL)
        def replacer(m):
            return m.group(1) + replace_with * len(m.group(2)) + m.group(3)
        return pattern.sub(replacer, text)
    else:
        think_string = re.compile(r'<think>(.*?)</think>', re.DOTALL).findall(text)

def process_jsonl(input_file, output_file, tok='llama'):
    with open(input_file, 'r') as infile:
        data = json.load(infile)
    
    if tok == 'llama':
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    else:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    output = []
    for item in data:
        if 'completion' in item:
            think_string = re.compile(r'<think>(.*?)</think>', re.DOTALL).findall(item['completion'])
            if len(think_string) == 0:
                continue
            think_string = think_string[0]
            think_tokenized = tokenizer(think_string)['input_ids']
            pause_tokens = ' .'.join(['' for _ in range(len(think_tokenized))])

            answer_string = re.compile(r'<answer>(.*?)</answer>', re.DOTALL).findall(item['completion'])
            if len(answer_string) == 0:
                continue
            answer_string = answer_string[0]

            modified_completion = f"<think>{pause_tokens}</think>\n<answer>{answer_string}</answer>"
            mc = tokenizer(modified_completion)['input_ids']
            ic = tokenizer(item['completion'])['input_ids']
            print(len(mc), len(ic))

            item['completion'] = modified_completion
            output.append(item)
            
    with open(output_file, 'w') as outfile:
        json.dump(output, outfile, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Process JSONL file to remove content within think tags')
    parser.add_argument('input_file', help='Input JSONL file path')
    parser.add_argument('output_file', help='Output JSONL file path')
    parser.add_argument('tokenizer', help='Tokenizer to use (llama or qwen)')
    
    args = parser.parse_args()

    try:
        process_jsonl(args.input_file, args.output_file, args.tokenizer)
        print(f"Successfully processed {args.input_file} and saved to {args.output_file}")
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
