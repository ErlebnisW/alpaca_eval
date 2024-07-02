export CUDA_VISIBLE_DEVICES=1
export HF_ENDPOINT="https://hf-mirror.com"

name=qwen2-alpaca_sp_3/sp_3

# GPT-4 evaluation
python3 -m alpaca_eval.generate \
    --model_name_or_path /data1/WM_workspace/MDSPO/output/qwen2-alpaca_sp_3/sp_3 \
    --output_dir /data1/WM_workspace/alpaca_eval/output/${name}
