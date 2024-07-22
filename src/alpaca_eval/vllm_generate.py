from __future__ import annotations
import argparse
import json
import os
from vllm import LLM, SamplingParams
import datasets
from rich.progress import track
import torch

PROMPT_BEGIN: str = 'BEGINNING OF CONVERSATION: '
PROMPT_USER: str = 'USER: {input} '
PROMPT_ASSISTANT: str = 'ASSISTANT:'
PROMPT_INPUT: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate models with vllm, utilizing all GPUs',
    )
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        help='The name or path of the model to load',
        required=True,
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/alpaca/',
        help='Where to store the evaluation output.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for generation',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Temperature for sampling',
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=2048,
        help='Maximum number of tokens to generate',
    )
    return parser.parse_args()

def extract(prompt: str) -> str:
    """Extract the original instruction from a constructed prompt."""
    # Remove PROMPT_BEGIN
    without_begin = prompt.replace(PROMPT_BEGIN, '', 1)
    # Remove PROMPT_USER
    without_user = without_begin.replace(PROMPT_USER, '', 1)
    # Remove PROMPT_ASSISTANT and everything after it
    instruction = without_user.split(PROMPT_ASSISTANT)[0]
    # Strip any leading/trailing whitespace
    return instruction.lstrip('USER:').strip()

def generate_answers(args, model_name_or_path: str, batch_size: int) -> list[dict]:
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")

    # Initialize LLM using all available GPUs
    with torch.no_grad():
        llm = LLM(model=model_name_or_path, tensor_parallel_size=num_gpus)
    
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    print(f'Generating answers with {model_name_or_path} using vllm')
    
    prompts = [PROMPT_INPUT.format(input=example["instruction"]) for example in eval_set]
    
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    
    results = []
    for i in track(range(0, len(prompts), batch_size), description="Generating..."):
        batch_prompts = prompts[i:i+batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
     
        for prompt, output in zip(batch_prompts, outputs):
            result = {
                "instruction": extract(prompt),  # Extract instruction part
                "output": output.outputs[0].text,
                "generator": model_name_or_path
            }
            results.append(result)
    
    return results

def main() -> None:
    """Main function."""
    args = parse_arguments()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True) 
    generate_file = os.path.join(output_dir, 'generated_answers.json')
    
    answers = generate_answers(args, args.model_name_or_path, args.batch_size)
    
    with open(generate_file, 'w', encoding='utf-8') as f:
        json.dump(answers, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()
