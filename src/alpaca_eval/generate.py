from __future__ import annotations

import argparse
import json
import os
import torch

from safe_rlhf.configs.constants import PROMPT_INPUT
from safe_rlhf.models import load_pretrained_models
from safe_rlhf.utils import to_device, log

import datasets

from rich.progress import track

def parse_arguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate models with alpaca eval',
    )

    # Model
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        help='the name or path of the first model (champion) in the arena to load from',
        required=True,
    )
    
    # Logging
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/alpaca/',
        help='Where to store the eval output.',
    )

    return parser.parse_args()


def generate_answers(model_name_or_path: str) -> list[dict]:
    model, tokenizer = load_pretrained_models(
        model_name_or_path,
        auto_device_mapping=True,
        trust_remote_code=True,
    )
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    log(f'Generating answers with {model_name_or_path}', 'yellow')
    
    results = []
    for example in track(eval_set, description='[red]Generating...'):
        prompt = PROMPT_INPUT.format(input=example["instruction"])
        input_ids = to_device(
            tokenizer(prompt, return_tensors='pt'),
            device=('cuda' if torch.cuda.is_available() else None),
        )
        output_ids = model.generate(
            **input_ids,
            max_length=2048,
        )
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)[len(prompt):]
        
        result = {
            "instruction": example["instruction"],
            "output": output,
            "generator": model_name_or_path
        }
        results.append(result)
    
    return results

def main() -> None:
    """The main function."""
    args = parse_arguments()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True) 
    generate_file = os.path.join(output_dir, 'generated_answers.json')
    
    answers = generate_answers(args.model_name_or_path)
    
    with open(generate_file, 'w', encoding='utf-8') as f:
        json.dump(answers, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()