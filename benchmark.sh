#!/bin/bash

# Function to print usage information
print_usage() {
    echo "Usage: $0 [-m <model-name>] [-i <tp-instances>] [-k <kv-cache>] [-v <vllm-version>]"
    echo "Example: $0 -m Qwen/Qwen2.5-Coder-7B-Instruct -i 3 -k 128 -v v0.6.5"
}

while getopts ":m:i:k:v:" opt; do
    case $opt in
        m) model=$OPTARG ;;
        i) inst=$OPTARG ;;
        k) kv=$OPTARG ;;
        v) vllm=$OPTARG ;;
        \?) echo "Invalid option: -$OPTARG" >&2; print_usage; exit 1 ;;
        :) echo "Option -$OPTARG requires an argument." >&2; print_usage; exit 1 ;;
    esac
done

# Set default values if not provided
model=${model:-Qwen/Qwen2.5-Coder-7B-Instruct}
kv=${kv:-128}
inst=${inst:-3}
vllm=${vllm:-v0.6.5}

cd scripts

git clone https://ray-project/llmperf || true
cd llmperf; pip install -e .; cd ..

# Define token sizes and concurrent users
declare -a input_tokens=(128 256 512 1024 2048 4096)
declare -a output_tokens=(128 256 512 1024 2048 4096)
declare -a concurrent_users=(1 10 50 100 150)

# Run benchmarks for input tokens varying while keeping output tokens = 1
for input in "${input_tokens[@]}"; do
    for users in "${concurrent_users[@]}"; do
        dir="benchmark/$vllm/$model/$inst/$kv/${input}_1_${users}"
        mkdir -p "$dir"
        OPENAI_API_BASE='http://localhost:4000/v1' OPENAI_API_KEY='1234' \
        python llmperf/token_benchmark_ray.py \
            --model $model \
            --mean-input-tokens $input \
            --stddev-input-tokens 6 \
            --mean-output-tokens 1 \
            --stddev-output-tokens 0 \
            --max-num-completed-requests 100 \
            --timeout 1000000 \
            --num-concurrent-requests $users \
            --results-dir "$dir" \
            --llm-api openai \
            --additional-sampling-params '{"max_tokens": "1", "ignore_eos": "True"}' \
            2>&1 | tee -a "$dir/run.log"
    done
done

# Run benchmarks for output tokens varying while keeping input tokens = 1
for output in "${output_tokens[@]}"; do
    for users in "${concurrent_users[@]}"; do
        dir="benchmark/$vllm/$model/$inst/$kv/1_${output}_${users}"
        mkdir -p "$dir"
        OPENAI_API_BASE='http://localhost:4000/v1' OPENAI_API_KEY='1234' \
        python llmperf/token_benchmark_ray.py \
            --model $model \
            --mean-input-tokens 1 \
            --stddev-input-tokens 0 \
            --mean-output-tokens $output \
            --stddev-output-tokens 6 \
            --max-num-completed-requests 100 \
            --timeout 1000000 \
            --num-concurrent-requests $users \
            --results-dir "$dir" \
            --llm-api openai \
            --additional-sampling-params '{"max_tokens": "'$output'", "ignore_eos": "True"}' \
            2>&1 | tee -a "$dir/run.log"
    done
done
