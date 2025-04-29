#!/usr/bin/env python3

"""
server-AMG-Model-Eval.py

Demonstrates:
  1) Running two TRT-LLM engines (FMHA ON / OFF), each with 200k context capacity.
  2) Each engine has a two-turn conversation:
       Turn 1 => Large 'book.rawtext' truncated to ~200k tokens
       Turn 2 => Same prefix + "write me the next chapter".
  3) After each Turn 2, we retrieve stats from llm.get_stats(), parse them,
     and compute TTFT, and other satistics about pre-fill.

** Warning ** 
200k contexts can require massive GPU memory or offload.
Make sure you're using Weka with TRT on our patch: https://github.com/NVIDIA/TensorRT-LLM/pull/3209
"""

import os
import json
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


###############################################################################
# Adjusted for 131k context
###############################################################################
SAFE_MAX_PROMPT_LEN = 131040 # <--- match new engine limit

def truncate_tokens(token_ids, max_len=SAFE_MAX_PROMPT_LEN, desc=""):
    """
    Truncates token_ids if they exceed max_len, printing a warning.
    """
    if len(token_ids) > max_len:
        print(f"Warning: {desc} token length {len(token_ids)} exceeds max {max_len}. Truncating.")
        return token_ids[:max_len]
    return token_ids


def convert_stats_to_string(stats_list):
    """
    Convert stats (list[dict]) from llm.get_stats() into a multi-line string.
    Each record line: "Record X: {...},"
    """
    lines = []
    lines.append("[\n")
    for idx, rec in enumerate(stats_list):
        rec_str = json.dumps(rec)  
        line = f"  Record {idx}: {rec_str},\n"
        lines.append(line)
    lines.append("]\n")
    return "".join(lines)


def manual_line_by_line_parser(raw_stats_string, engine_name="UNKNOWN"):
    """
    Naive parser for the multi-line string from convert_stats_to_string().
    Then compute TTFT by summing iterLatencyMS until we detect decode
    (avgNumDecodedTokensPerIter>0.0 or numGenRequests>0).
    """
    print(f"\n>>> ENTERING manual_line_by_line_parser() for {engine_name} <<<")
    lines = raw_stats_string.splitlines()

    reconstructed = []
    for line_idx, line in enumerate(lines):
        line_str = line.strip()
        print(f"[DEBUG] {engine_name} parser line[{line_idx}]: {line_str}")

        if line_str.startswith("Record "):
            colon_pos = line_str.find(":")
            if colon_pos < 0:
                continue
            dict_part = line_str[colon_pos + 1:].strip()
            if dict_part.endswith(","):
                dict_part = dict_part[:-1].strip()
            try:
                parsed = json.loads(dict_part)
                reconstructed.append(parsed)
            except Exception as e:
                print(f"[DEBUG] {engine_name} - JSON parse error on line: {line_str}\nError: {e}")

    print(f"[DEBUG] {engine_name} parser => Reconstructed {len(reconstructed)} records.")

    # TTFT logic
    cumulative_ms = 0.0
    found_decode = False
    prefill_count = 0
    decode_count = 0
    ttft_ms = None

    for idx, rec in enumerate(reconstructed):
        lat_ms = rec.get("iterLatencyMS", 0.0)
        inflight = rec.get("inflightBatchingStats", {})
        avg_decoded = inflight.get("avgNumDecodedTokensPerIter", 0.0)
        num_gen = inflight.get("numGenRequests", 0)

        cumulative_ms += lat_ms

        if not found_decode:
            if (avg_decoded > 0.0) or (num_gen > 0):
                ttft_ms = cumulative_ms
                found_decode = True
                decode_count += 1
            else:
                prefill_count += 1
        else:
            decode_count += 1

    result = {
        "records": reconstructed,
        "TTFT_ms": ttft_ms,
        "prefill_iterations": prefill_count,
        "decode_iterations": decode_count
    }

    print(f"[DEBUG] {engine_name} parser => FINAL parsed result: {result}")
    print(f">>> LEAVING manual_line_by_line_parser() for {engine_name} <<<\n")
    return result


@click.command()
def main():
    """
    Evaluate 2-turn conversation on two TRT-LLM engines, each capable of 200k context.
    """
    # Updated engine paths for 200k context
    engine1_path = "/mnt/weka/Models/Engines/llama-3.1-8b-engine-131k-FMHA-ON"
    engine2_path = "/mnt/weka/Models/Engines/llama-3.1-8b-engine-131k-FMHA-OFF"


    tokenizer_path = "/mnt/weka/Models/Safetensors/Meta-Llama-3.1-8B-Instruct"

    # Set build_config to 200k, so the LLM object matches engine capacity
    build_config = BuildConfig(
        max_beam_width=1,
        max_batch_size=8,
        max_num_tokens=200000,  # 200k
        max_seq_len=200000      # 200k
    )

    def create_llm(engine_path, policy, block_reuse):
        kv_cache_config = KvCacheConfig(
            enable_block_reuse=block_reuse,
            host_cache_size=45000000000,
            max_tokens=200000,  # caution: set high for large contexts
            secondary_offload_min_priority=30
        )
        scheduler_config = SchedulerConfig(capacity_scheduler_policy=policy)

        llm_args = LlmArgs.from_kwargs(
            model=engine_path,
            tokenizer=tokenizer_path,
            backend=None,
            trust_remote_code=False,
            build_config=build_config,
            kv_cache_config=kv_cache_config,
            scheduler_config=scheduler_config,
            tensor_parallel_size=8,
            pipeline_parallel_size=1
        )
        return LLM(**llm_args.to_dict())

    # Retention config for FMHA=ON (Engine 1), none for Engine 2
    retention1 = trtllm.KvCacheRetentionConfig(
        [
            trtllm.KvCacheRetentionConfig.TokenRangeRetentionConfig(
                token_start=0,
                token_end=None,
                priority=30,
                duration_ms=datetime.timedelta(seconds=30)
            )
        ],
        decode_retention_priority=80
    )
    retention2 = None

    # Load large text
    rawtext_file = "/code/tensorrt_llm/book.rawtext"
    if not os.path.exists(rawtext_file):
        raise FileNotFoundError(f"Could not find {rawtext_file}")

    with open(rawtext_file, "r", encoding="utf-8") as f:
        book_text = f.read()

    # ENGINE 1 => FMHA=ON
    engine1 = create_llm(
        engine1_path,
        policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        block_reuse=True
    )
    print("\n====== ENGINE 1: FMHA=ON, 200k ======\n")

    # Turn 1
    turn1_ids = engine1.tokenizer.encode(book_text)
    turn1_ids = truncate_tokens(turn1_ids, desc="ENGINE1 Turn 1")
    print(f"[ENGINE1] Turn 1 => {len(turn1_ids)} tokens")
    result1 = engine1.generate(inputs=turn1_ids, kv_cache_retention_config=retention1)

    # Turn 2
    turn2_text = book_text + "\n\nwrite me the next chapter"
    turn2_ids = engine1.tokenizer.encode(turn2_text)
    turn2_ids = truncate_tokens(turn2_ids, desc="ENGINE1 Turn 2")
    print(f"[ENGINE1] Turn 2 => {len(turn2_ids)} tokens")

    old_stats_1 = engine1.get_stats()
    old_len_1 = len(old_stats_1)

    result2 = engine1.generate(inputs=turn2_ids, kv_cache_retention_config=retention1)
    new_stats_1 = engine1.get_stats()
    after_len_1 = len(new_stats_1)
    print(f"[ENGINE1] old_stats length={old_len_1}, new_stats length={after_len_1}")

    if after_len_1 > old_len_1:
        turn2_stats_1 = new_stats_1[old_len_1:]
    else:
        turn2_stats_1 = new_stats_1

    raw_str_1 = convert_stats_to_string(turn2_stats_1)
    print("[ENGINE1] RAW Turn 2 stats =>\n", raw_str_1)
    parse_res_1 = manual_line_by_line_parser(raw_str_1, "ENGINE1-FMHA-ON")

    finish_reason1 = "NoOutput"
    if result2.outputs and len(result2.outputs) > 0:
        finish_reason1 = result2.outputs[0].finish_reason
    print(f"[ENGINE1] Turn2 finish => {finish_reason1}")
    print("[ENGINE1] TTFT_ms =", parse_res_1["TTFT_ms"])
    print("====================================\n")

    # ENGINE 2 => FMHA=OFF
    engine2 = create_llm(
        engine2_path,
        policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        block_reuse=False
    )
    print("\n====== ENGINE 2: FMHA=OFF, 200k ======\n")

    # Turn 1
    turn1_ids2 = engine2.tokenizer.encode(book_text)
    turn1_ids2 = truncate_tokens(turn1_ids2, desc="ENGINE2 Turn 1")
    print(f"[ENGINE2] Turn 1 => {len(turn1_ids2)} tokens")
    result1_e2 = engine2.generate(inputs=turn1_ids2, kv_cache_retention_config=retention2)

    # Turn 2
    turn2_text2 = book_text + "\n\nwrite me the next chapter"
    turn2_ids2 = engine2.tokenizer.encode(turn2_text2)
    turn2_ids2 = truncate_tokens(turn2_ids2, desc="ENGINE2 Turn 2")
    print(f"[ENGINE2] Turn 2 => {len(turn2_ids2)} tokens")

    old_stats_2 = engine2.get_stats()
    old_len_2 = len(old_stats_2)

    result2_e2 = engine2.generate(inputs=turn2_ids2, kv_cache_retention_config=retention2)
    new_stats_2 = engine2.get_stats()
    after_len_2 = len(new_stats_2)
    print(f"[ENGINE2] old_stats length={old_len_2}, new_stats length={after_len_2}")

    if after_len_2 > old_len_2:
        turn2_stats_2 = new_stats_2[old_len_2:]
    else:
        turn2_stats_2 = new_stats_2

    raw_str_2 = convert_stats_to_string(turn2_stats_2)
    print("[ENGINE2] RAW Turn 2 stats =>\n", raw_str_2)
    parse_res_2 = manual_line_by_line_parser(raw_str_2, "ENGINE2-FMHA-OFF")

    finish_reason2 = "NoOutput"
    if result2_e2.outputs and len(result2_e2.outputs) > 0:
        finish_reason2 = result2_e2.outputs[0].finish_reason
    print(f"[ENGINE2] Turn2 finish => {finish_reason2}")
    print("[ENGINE2] TTFT_ms =", parse_res_2["TTFT_ms"])
    print("====================================\n")

    # Comparison
    print("==== COMPARISON RESULTS ====")
    print("ENGINE1-FMHA-ON => TTFT:", parse_res_1["TTFT_ms"])
    print("ENGINE2-FMHA-OFF => TTFT:", parse_res_2["TTFT_ms"])
    ttft1 = parse_res_1["TTFT_ms"]
    ttft2 = parse_res_2["TTFT_ms"]

    if ttft1 and ttft2 and ttft1>0 and ttft2>0:
        diff_pct = (ttft2 - ttft1)/ttft1*100.0
        print(f"TTFT difference (Engine2 vs Engine1): {diff_pct:.2f}%")
    else:
        print("Could not compute TTFT difference (None or zero).")

    print("==== END COMPARISON ====\n")


if __name__ == "__main__":
    main()
