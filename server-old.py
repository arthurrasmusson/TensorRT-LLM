import click
import datetime
import numpy as np

import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm.llmapi import (
    LLM,
    BuildConfig,
    KvCacheConfig,
    SchedulerConfig,
    CapacitySchedulerPolicy,
)
from tensorrt_llm.bindings.executor import DynamicBatchConfig
from tensorrt_llm.llmapi.llm_utils import LlmArgs


@click.command()
@click.option("--model",
              type=str,
              default="/mnt/weka/Models/Engines/llama-3.1-8b-engine-2ec70d79308f2e445031393703e5be5793fd777a-singleengine",
              #default="/mnt/weka/Models/Engines/Llama-3.1-405B-INT8-TP8-PP2",
              help="Path to the TRT-LLM engine folder or HF checkpoint.")
@click.option("--tokenizer",
              type=str,
              default="/mnt/weka/Models/Safetensors/Meta-Llama-3.1-8B-Instruct",
              #default="/mnt/weka/Models/Safetensors/Llama-3.1-405B",
              help="Path or name of the tokenizer. "
                   "If using a TRT engine with built-in tokenizer, this can be omitted.")
@click.option("--backend",
              type=click.Choice(["pytorch"]),
              default=None,
              help="Set to 'pytorch' if using a PyTorch HF model; otherwise, TRT-C++ is used.")
@click.option("--max_beam_width",
              type=int,
              default=BuildConfig.max_beam_width,
              help="Maximum beam width for decoding.")
@click.option("--max_batch_size",
              type=int,
              default=BuildConfig.max_batch_size,
              help="Max requests that the engine can schedule at once.")
@click.option("--max_num_tokens",
              type=int,
              default=BuildConfig.max_num_tokens,
              help="Max number of *input* tokens per batch (after padding removal).")
@click.option("--max_seq_len",
              type=int,
              default=BuildConfig.max_seq_len,
              help="Max total length (prompt + generated output).")
@click.option("--tp_size", type=int, default=1, help="Tensor parallel size.")
@click.option("--pp_size", type=int, default=1, help="Pipeline parallel size.")
def main(model,
         tokenizer,
         backend,
         max_beam_width,
         max_batch_size,
         max_num_tokens,
         max_seq_len,
         tp_size,
         pp_size):
    """
    Continuously submit random input tokens to the LLM.
    """

    # -------------------------------------------------------------------------
    # 1. Build configuration objects
    # -------------------------------------------------------------------------
    build_config = BuildConfig(
        max_beam_width=max_beam_width,
        max_batch_size=max_batch_size,
        max_num_tokens=max_num_tokens,
        max_seq_len=max_seq_len
    )

    kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        host_cache_size=45000000000,  # 45 GiB for offloading
        max_tokens=8000,
        secondary_offload_min_priority=30
    )

    dynamic_batch_config = DynamicBatchConfig(
        enable_batch_size_tuning=True,
        enable_max_num_tokens_tuning=False,
        dynamic_batch_moving_average_window=128
    )

    scheduler_config = SchedulerConfig(
        capacity_scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION
    )

    # -------------------------------------------------------------------------
    # 2. Create LLM object (wrapped Executor internally)
    # -------------------------------------------------------------------------
    llm_args = LlmArgs.from_kwargs(
        model=model,
        tokenizer=tokenizer,
        backend=backend,
        trust_remote_code=False,  # set True if you're using HF models requiring remote code
        build_config=build_config,
        kv_cache_config=kv_cache_config,
        # Optionally enable dynamic_batch_config here:
        # dynamic_batch_config=dynamic_batch_config,
        scheduler_config=scheduler_config,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size
    )

    llm = LLM(**llm_args.to_dict())

    # -------------------------------------------------------------------------
    # 3. Construct the KvCacheRetentionConfig
    # -------------------------------------------------------------------------
    retention_config = trtllm.KvCacheRetentionConfig(
        [
            trtllm.KvCacheRetentionConfig.TokenRangeRetentionConfig(
                0, None, 30, datetime.timedelta(seconds=30)
            )
        ],
        80  # Retention Priority for newly generated tokens
    )

    #perf_metrics = trtllm.RequestPerfMetrics
    # -------------------------------------------------------------------------
    # 4. Continuously generate
    # -------------------------------------------------------------------------
    prompt_count = 0
    while True:
        # Generate random input tokens
        input_token_ids = np.random.randint(0, 1000, size=320).tolist()

        # Perform inference
        result = llm.generate(
            inputs=input_token_ids,
            kv_cache_retention_config=retention_config
        )

        prompt_count += 1
        print("[SERVER.PY] PROMPT ITERATION: ", prompt_count)
        if prompt_count % 50 == 0:
            print("Entered 50th prompt - getting stats")
            # Retrieve inference metrics from the List[dict] returned by get_stats()
            iteration_stats_result = llm.get_stats()
            print("Got stats")
            print(f"Collected iteration stats at prompt #{prompt_count}:")
            for idx, stat in enumerate(iteration_stats_result):
                print(f"  Iteration {idx} stats: {stat}")

            # Exit after printing stats once
            print("Exiting after printing stats.")
            break

        # 5. Print results
        if not result.outputs:
            print("No generation outputs returned.")
            continue

        generated_ids = result.outputs[0].token_ids
        finish_reason = result.outputs[0].finish_reason
        print("Generated tokens:", generated_ids)
        print("Finish reason:", finish_reason)
        print("-" * 50)


if __name__ == "__main__":
    main()

