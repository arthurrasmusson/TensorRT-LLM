import copy
import json
import os
import time
import torch.multiprocessing as mp
import shutil
import argparse

from vllm.outputs import RequestOutput
from vllm.sequence import RequestMetrics

from vllm.config import KVTransferConfig
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from lmcache.experimental.cache_engine import LMCacheEngineBuilder
from lmcache.integration.vllm.utils import ENGINE_NAME

# Model configuration
#model_name = "neuralmagic/Meta-Llama-3.1-405B-Instruct-quantized.w4a16"
model_name = "meta-llama/Llama-3.1-405B-Instruct-FP8"
context_file = 'book.rawtext'
output_file = "offline_inference_outputs.jsonl"

# Read the context from file
with open(context_file, 'r') as f:
    context_text = f.read()
assert context_text is not None

tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_context_text(tokenizer, context_t, max_model_len):
    """Truncate the context if it exceeds max_model_len - 300 tokens."""
    context_tokens = tokenizer.encode(context_t)
    if len(context_tokens) > max_model_len:
        context_t = tokenizer.decode(context_tokens[:max_model_len - 300])
    return context_t

def get_context_length(tokenizer, context_messages):
    """Compute how many characters the chat template has (rough approximation)."""
    return len(tokenizer.apply_chat_template(context_messages, tokenize=False))

def gen_prompts(tokenizer, context_messages, user_inputs_of_batch):
    """Generate full prompt strings by appending each user input to the base context messages."""
    generated_prompts = []
    for user_input in user_inputs_of_batch:
        copied_context_messages = copy.deepcopy(context_messages)
        copied_context_messages.append({"role": "user", "content": user_input})
        generated_prompts.append(tokenizer.apply_chat_template(copied_context_messages, tokenize=False))
    return generated_prompts

def append_outputs(output_file_name, outputs, context_length, ttft):
    """Write the output (prompt and generation) plus TTFT to offline_inference_outputs.jsonl."""
    user_inputs = []
    generated_texts = []
    for output in outputs:
        prompt = output.prompt
        user_input = prompt[context_length:]
        user_inputs.append(user_input)
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text)

    json_dict = {
        "user_inputs": user_inputs,
        "generated_texts": generated_texts,
        "ttft_seconds": ttft
    }
    with open(output_file_name, "a") as f:
        f.write(json.dumps(json_dict) + '\n')

def clear_directory(directory_path):
    # Ensure the directory exists
    if os.path.exists(directory_path):
        # Iterate over all the entries in the directory
        for entry in os.listdir(directory_path):
            entry_path = os.path.join(directory_path, entry)
            try:
                # Remove files
                if os.path.isfile(entry_path) or os.path.islink(entry_path):
                    os.remove(entry_path)
                # Remove directories and their contents
                elif os.path.isdir(entry_path):
                    shutil.rmtree(entry_path)
            except Exception as e:
                print(f'Failed to delete {entry_path}. Reason: {e}')
    else:
        print(f'The directory {directory_path} does not exist.')

def log_benchmark(benchmark_file, context_size, run_index, ttft):
    """Log each run's TTFT to the benchmark file."""
    benchmark_entry = {
        "run_index": run_index,
        "context_size": context_size,
        "ttft_seconds": ttft
    }
    with open(benchmark_file, "a") as f:
        f.write(json.dumps(benchmark_entry) + '\n')

def get_TTFT(output: list[RequestOutput]):
    """Time to First Token (TTFT) in vLLM metrics."""
    # In this script, we only request 1 prompt at a time, so output[0] is the relevant item.
    metrics: RequestMetrics = output[0].metrics
    return metrics.first_token_time - metrics.arrival_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run offline inference with a specified benchmark file.")
    parser.add_argument(
        "--benchmark_file", 
        type=str, 
        default="benchmark_results.jsonl", 
        help="Name of the benchmark file to write results."
    )
    parser.add_argument(
        "--context_size", 
        type=int, 
        default=131072, 
        help="Size of the context window."
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=8,
        help="Number of tensor parallel chunks."
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=2,  # Default to 2
        help="Number of times to repeat the inference measurements."
    )

    args = parser.parse_args()

    benchmark_file = args.benchmark_file
    context_size = args.context_size
    tensor_parallel_size = args.tensor_parallel_size
    num_runs = args.num_runs

    # Set multi-processing method
    mp.set_start_method('spawn', force=True)

    # Clear out old data in the outputs
    with open(output_file, "w") as f:
        pass
    with open(benchmark_file, "w") as f:
        pass

    # Prepare the LLM
    sampling_params_generation = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=1024)
    ktc = KVTransferConfig.from_cli('{"kv_connector":"LMCacheConnector", "kv_role":"kv_both"}')
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.80,
        enable_chunked_prefill=False,
        kv_transfer_config=ktc,
        max_model_len=128000,
        tensor_parallel_size=tensor_parallel_size
    )

    print(f"Running inference with context size: {context_size}, for {num_runs} run(s).")

    # Prepare a short context and prompt
    context_text_trimmed = get_context_text(tokenizer, context_text, context_size)
    context_messages = [
        {"role": "user", "content": f"I've got a document:```\n{context_text_trimmed}\n```."},
        {"role": "assistant", "content": "I've got your document"}
    ]
    user_inputs_batch = ["Give me a concise summary of the story."]
    context_length = get_context_length(tokenizer, context_messages)
    prompts = gen_prompts(tokenizer, context_messages, user_inputs_batch)

    # Run the benchmark multiple times
    for run_idx in range(1, num_runs + 1):
        print(f"\n===== Run #{run_idx} =====")
        outputs = llm.generate(prompts, sampling_params_generation)
        ttft = get_TTFT(outputs)
        print(f"[Run #{run_idx}] TTFT: {ttft:.4f}s")

        # Record outputs and TTFT
        append_outputs(output_file, outputs, context_length, ttft)
        log_benchmark(benchmark_file, context_size, run_idx, ttft)

    # clear the directory
    clear_directory("/mnt/weka/4d0r/lmcache-exp/")

    # Clean up LMCache engine
    LMCacheEngineBuilder.destroy(ENGINE_NAME)
    time.sleep(5)  # Small delay to ensure the engine shuts down
