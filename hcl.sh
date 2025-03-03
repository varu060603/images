#!/bin/bash

# Function to print usage information
print_usage() {
    echo "Usage: $0 [-m <model-name>] [-i <tp-instances>] [-k <kv-cache>] [-v <vllm-version>]"
    echo "Example: $0 -m Qwen/Qwen2.5-Coder-7B-Instruct -i 3 -k 128 -v v0.6.5"
}

# Parse command-line options
while getopts ":m:i:k:v:" opt; do
    case $opt in
        m)
            model=$OPTARG
            ;;
        i)
            inst=$OPTARG
            ;;
        k)
            kv=$OPTARG
            ;;
        v)
            vllm=$OPTARG
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            print_usage
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            print_usage
            exit 1
            ;;
    esac
done

# Set default values if not provided
if [ -z "$model" ]; then
    model="Qwen/Qwen2.5-Coder-7B-Instruct"
fi

if [ -z "$kv" ]; then
    kv=128
fi

if [ -z "$inst" ]; then
    inst=3
fi

if [ -z "$vllm" ]; then
    vllm="v0.6.5"
fi

# Change to scripts directory
cd scripts || exit 1

# Install llmperf (assuming this is still needed)
if [ ! -d "llmperf" ]; then
    git clone https://github.com/ray-project/llmperf.git
    cd llmperf
    pip install -e .
    cd ..
fi

# Benchmark with 32 input tokens, 550 output tokens, and varying concurrent requests

# Concurrent requests: 10
mkdir -p out/$vllm/$model/$inst/$kv/32_550_10
OPENAI_API_BASE='http://localhost:4000/v1' OPENAI_API_KEY='1234' python llmperf/token_benchmark_ray.py \
    --model "$model" \
    --mean-input-tokens 32 \
    --stddev-input-tokens 6 \
    --mean-output-tokens 550 \
    --stddev-output-tokens 102 \
    --max-num-completed-requests 100 \
    --timeout 10000 \
    --num-concurrent-requests 10 \
    --results-dir out/$vllm/$model/$inst/$kv/32_550_10 \
    --llm-api openai \
    --additional-sampling-params '{"max_tokens": "550", "ignore_eos": "True"}' 2>&1 | tee -a out/$vllm/$model/$inst/$kv/32_550_10/run.log

# Concurrent requests: 30
mkdir -p out/$vllm/$model/$inst/$kv/32_550_30
OPENAI_API_BASE='http://localhost:4000/v1' OPENAI_API_KEY='1234' python llmperf/token_benchmark_ray.py \
    --model "$model" \
    --mean-input-tokens 32 \
    --stddev-input-tokens 6 \
    --mean-output-tokens 550 \
    --stddev-output-tokens 102 \
    --max-num-completed-requests 300 \
    --timeout 10000 \
    --num-concurrent-requests 30 \
    --results-dir out/$vllm/$model/$inst/$kv/32_550_30 \
    --llm-api openai \
    --additional-sampling-params '{"max_tokens": "550", "ignore_eos": "True"}' 2>&1 | tee -a out/$vllm/$model/$inst/$kv/32_550_30/run.log

# Concurrent requests: 100
mkdir -p out/$vllm/$model/$inst/$kv/32_550_100
OPENAI_API_BASE='http://localhost:4000/v1' OPENAI_API_KEY='1234' python llmperf/token_benchmark_ray.py \
    --model "$model" \
    --mean-input-tokens 32 \
    --stddev-input-tokens 6 \
    --mean-output-tokens 550 \
    --stddev-output-tokens 102 \
    --max-num-completed-requests 1000 \
    --timeout 10000 \
    --num-concurrent-requests 100 \
    --results-dir out/$vllm/$model/$inst/$kv/32_550_100 \
    --llm-api openai \
    --additional-sampling-params '{"max_tokens": "550", "ignore_eos": "True"}' 2>&1 | tee -a out/$vllm/$model/$inst/$kv/32_550_100/run.log
