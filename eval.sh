export HF_ENDPOINT="https://hf-mirror.com"
export OPENAI_CLIENT_CONFIG_PATH="/data1/WM_workspace/alpaca_eval/client_configs/custom.yaml"

alpaca_eval --model_outputs /data1/WM_workspace/alpaca_eval/output/qwen2-alpaca_sp_3/sp_3/generated_answers.json \
            --annotators_config claude_3_opus_ranking \
