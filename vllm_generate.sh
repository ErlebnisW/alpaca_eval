export CUDA_VISIBLE_DEVICES=1
export HF_ENDPOINT="https://hf-mirror.com"

name=gemma-2b_alpaca_sft

# GPT-4 evaluation
python3 -m alpaca_eval.vllm_generate \
    --model_name_or_path /data1/WM_workspace/MDSPO/output/fix_temp_1.0/ \
    --output_dir /data1/WM_workspace/alpaca_eval/output/${name} \
    --temperature 0.7 \
    --max_tokens 2048 \
    --batch_size 16
