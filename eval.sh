export HF_ENDPOINT="https://hf-mirror.com"
export OPENAI_CLIENT_CONFIG_PATH="/data1/WM_workspace/alpaca_eval/client_configs/deepseek.yaml"

name=gemma_2b_alpaca_sft
alpaca_eval --model_outputs /data1/WM_workspace/alpaca_eval/output/fix_temp_1/generated_answers.json \
            --annotators_config deepseek-chat \
            --output_path /data1/WM_workspace/alpaca_eval/output/results/${name} \
            # --reference_outputs /data1/WM_workspace/alpaca_eval/output/gemma-2b_alpaca_sft/generated_answers.json
