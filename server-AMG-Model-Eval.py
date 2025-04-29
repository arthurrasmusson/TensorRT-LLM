#!/usr/bin/env python3

"""
server-AMG-Model-Eval.py

Demonstrates:
  1) Running two TRT-LLM engines (ENGINE1-FMHA-ON and ENGINE2-FMHA-OFF).
  2) Each engine has a two-turn conversation:
       Turn 1: Large 'book.rawtext' truncated to ~7936 tokens
       Turn 2: Same truncated prefix plus "write me the next chapter".
  3) After each Turn 2, we:
     - Retrieve stats from llm.get_stats() (list[dict]).
     - Convert them to a raw multi-line string for debugging.
     - Print that raw string in full.
     - Parse it line-by-line with a "manual_line_by_line_parser" function
       that looks for 'iterLatencyMS', 'avgNumDecodedTokensPerIter',
       'numGenRequests', etc.
     - Print the resulting TTFT, prefill_iterations, decode_iterations, etc.

** Special Debugging Focus **
We've added *extremely* verbose debug prints around stats conversion to string
and the manual parser to ensure we see exactly what data is being handled.

** Example Usage **
    python3 server-AMG-Model-Eval.py
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
# Constants
###############################################################################
SAFE_MAX_PROMPT_LEN = 7936

###############################################################################
# Stats Conversion and Manual Parser
###############################################################################
def convert_stats_to_string(stats_list):
    """
    Convert the list[dict] returned by llm.get_stats() into a single
    multi-line string. Each record is prefixed by "Record X:".
    """
    lines = []
    lines.append("[\n")
    for idx, rec in enumerate(stats_list):
        # Convert the dict to a naive pseudo-JSON string:
        # (We can do an actual JSON dump, but let's do it manually for demonstration.)
        rec_str = json.dumps(rec)  # Produces a valid JSON representation
        line = f"  Record {idx}: {rec_str},\n"
        lines.append(line)
    lines.append("]\n")
    return "".join(lines)

def manual_line_by_line_parser(raw_stats_string, engine_name="UNKNOWN"):
    """
    Very naive line-by-line parser that looks for lines starting with "Record X:".
    Then tries to parse the JSON dict portion with json.loads().

    Once we reconstruct the list of records, we do a TTFT calculation:
      - Keep summing iterLatencyMS
      - We define decode as: avgNumDecodedTokensPerIter>0.0 or numGenRequests>0

    Returns a dict: {
       "records": <list_of_dicts>,
       "TTFT_ms": ...,
       "prefill_iterations": ...,
       "decode_iterations": ...,
    }
    """
    print(f"\n>>> ENTERING manual_line_by_line_parser() for {engine_name} <<<")
    lines = raw_stats_string.splitlines()

    reconstructed = []
    # We'll parse each line that starts with "Record X:"
    for line_idx, line in enumerate(lines):
        line_str = line.strip()
        print(f"[DEBUG] {engine_name} parser line[{line_idx}]: {line_str}")

        if line_str.startswith("Record "):
            # example: Record 0: {"iter":32,"iterLatencyMS":...},
            # we want the JSON part after the colon
            colon_pos = line_str.find(":")
            if colon_pos < 0:
                continue  # skip
            dict_part = line_str[colon_pos+1:].strip()
            # remove trailing comma if present
            if dict_part.endswith(","):
                dict_part = dict_part[:-1].strip()

            # Now dict_part should be valid JSON because we used json.dumps earlier
            try:
                parsed = json.loads(dict_part)
                reconstructed.append(parsed)
            except Exception as e:
                print(f"[DEBUG] {engine_name} - Could not parse JSON for line: {line_str}\nError: {e}")

    print(f"[DEBUG] {engine_name} parser => Reconstructed {len(reconstructed)} records.")

    # Now do TTFT logic
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
        "decode_iterations": decode_count,
    }

    print(f"[DEBUG] {engine_name} parser => FINAL parsed result: {result}")
    print(f">>> LEAVING manual_line_by_line_parser() for {engine_name} <<<\n")
    return result

###############################################################################
# Helper to truncate tokens
###############################################################################
def truncate_tokens(token_ids, max_len=SAFE_MAX_PROMPT_LEN, desc=""):
    if len(token_ids) > max_len:
        print(f"Warning: {desc} token length {len(token_ids)} exceeds max {max_len}. Truncating.")
        return token_ids[:max_len]
    return token_ids

###############################################################################
# The main script
###############################################################################
@click.command()
def main():
    # Engine paths
    engine1_path = "/mnt/weka/Models/Engines/llama-3.1-8b-engine-EVAL-FMHA-ON"
    engine2_path = "/mnt/weka/Models/Engines/llama-3.1-8b-engine-EVAL-FMHA-OFF"
    tokenizer_path = "/mnt/weka/Models/Safetensors/Meta-Llama-3.1-8B-Instruct"

    # Build config for both
    build_config = BuildConfig(
        max_beam_width=1,
        max_batch_size=8,
        max_num_tokens=2048,
        max_seq_len=8192
    )
    # We'll vary the capacity_scheduler_policy and enable_block_reuse per engine
    # Engine1 => FMHA=ON => capacity=MAX_UTILIZATION => block_reuse=True
    # Engine2 => FMHA=OFF => capacity=GUARANTEED_NO_EVICT => block_reuse=False

    def create_llm(engine_path, tokenizer_path, policy, block_reuse, tp_size=8, pp_size=1):
        kv_cache_config = KvCacheConfig(
            enable_block_reuse=block_reuse,
            host_cache_size=45_000_000_000,
            max_tokens=8000,
            secondary_offload_min_priority=30
        )
        scheduler_config = SchedulerConfig(
            capacity_scheduler_policy=policy
        )
        llm_args = LlmArgs.from_kwargs(
            model=engine_path,
            tokenizer=tokenizer_path,
            backend=None,
            trust_remote_code=False,
            build_config=build_config,
            kv_cache_config=kv_cache_config,
            scheduler_config=scheduler_config,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size
        )
        return LLM(**llm_args.to_dict())

    # Retention config for engine1, none for engine2
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

    # Load the big book file
    rawtext_file = "/code/tensorrt_llm/book.rawtext"
    if not os.path.exists(rawtext_file):
        raise FileNotFoundError(f"Cannot find {rawtext_file}")

    with open(rawtext_file, "r", encoding="utf-8") as f:
        book_text = f.read()

    #----------------------------------------------------------------
    # ENGINE 1: FMHA=ON
    #----------------------------------------------------------------
    engine1 = create_llm(
        engine1_path, tokenizer_path,
        policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        block_reuse=True
    )
    print("\n====================================================")
    print("Running ENGINE1-FMHA-ON (WEKA AUGMENTED MEMORY GRID ON)")
    print("====================================================\n")

    # Turn 1
    turn1_ids = engine1.tokenizer.encode(book_text)
    turn1_ids = truncate_tokens(turn1_ids, desc="Turn 1 Prompt")
    print(f"[ENGINE1-FMHA-ON] Turn 1 => {len(turn1_ids)} tokens")

    result1 = engine1.generate(
        inputs=turn1_ids,
        kv_cache_retention_config=retention1
    )
    # (We won't parse Turn 1 stats in detail here, but you could if you want.)
    # Turn 2
    turn2_text = book_text + "\n\nwrite me the next chapter"
    turn2_ids = engine1.tokenizer.encode(turn2_text)
    turn2_ids = truncate_tokens(turn2_ids, desc="Turn 2 Prompt")
    print(f"[ENGINE1-FMHA-ON] Turn 2 => {len(turn2_ids)} tokens")

    # old stats length
    old_stats_1 = engine1.get_stats()
    old_len_1 = len(old_stats_1)

    # manually print what we need to convert to a single large string for old_stats_1:
    for idx, stat in enumerate(old_stats_1):
        print(f" old_stats_1 Iteration {idx} stats: {stat}")

    # generate
    result2 = engine1.generate(
        inputs=turn2_ids,
        kv_cache_retention_config=retention1
    )
    # new stats
    new_stats_1 = engine1.get_stats()
    after_len_1 = len(new_stats_1)

    # manually print what we need to convert to a single large string for new_stats_1:
    for idx, stat in enumerate(new_stats_1):
        print(f" new_stats_1 Iteration {idx} stats: {stat}")

    print(f"[ENGINE1-FMHA-ON] Stats right AFTER Turn 2 => length={after_len_1}")
    # We only want the newly appended stats from old_len_1..end
    turn2_stats_list = new_stats_1[old_len_1:]
    # Convert to string
    raw_str_engine1 = convert_stats_to_string(turn2_stats_list)
    # Print the string
    print("[ENGINE1-FMHA-ON] => RAW string of Turn 2 stats:\n", raw_str_engine1)
    # Parse
    parse_res_1 = manual_line_by_line_parser(raw_str_engine1, engine_name="ENGINE1-FMHA-ON")

    # Print parse results
    print(f"[ENGINE1-FMHA-ON] PARSED Turn 2 Stats => TTFT_ms={parse_res_1['TTFT_ms']}, "
          f"prefill={parse_res_1['prefill_iterations']}, decode={parse_res_1['decode_iterations']}")
    finish_reason_2 = "No Output"
    if result2.outputs:
        finish_reason_2 = result2.outputs[0].finish_reason
    print(f"[ENGINE1-FMHA-ON] Turn 2 finish => {finish_reason_2}")
    print("====================================================\n")

    #----------------------------------------------------------------
    # ENGINE 2: FMHA=OFF
    #----------------------------------------------------------------
    engine2 = create_llm(
        engine2_path, tokenizer_path,
        policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        block_reuse=False
    )
    print("\n====================================================")
    print("Running ENGINE2-FMHA-OFF (WEKA AUGMENTED MEMORY GRID OFF)")
    print("====================================================\n")

    # Turn 1
    turn1_ids_2 = engine2.tokenizer.encode(book_text)
    turn1_ids_2 = truncate_tokens(turn1_ids_2, desc="Turn 1 Prompt")
    print(f"[ENGINE2-FMHA-OFF] Turn 1 => {len(turn1_ids_2)} tokens")

    result1_e2 = engine2.generate(
        inputs=turn1_ids_2,
        kv_cache_retention_config=retention2
    )
    # Turn 2
    turn2_text_2 = book_text + "\n\nwrite me the next chapter"
    turn2_ids_2 = engine2.tokenizer.encode(turn2_text_2)
    turn2_ids_2 = truncate_tokens(turn2_ids_2, desc="Turn 2 Prompt")
    print(f"[ENGINE2-FMHA-OFF] Turn 2 => {len(turn2_ids_2)} tokens")

    old_stats_2 = engine2.get_stats()
    old_len_2 = len(old_stats_2)

    # manually print what we need to convert to a single large string for old_stats_2:
    for idx, stat in enumerate(old_stats_2):
        print(f"  Iteration {idx} stats: {stat}")

    result2_e2 = engine2.generate(
        inputs=turn2_ids_2,
        kv_cache_retention_config=retention2
    )
    new_stats_2 = engine2.get_stats()
    after_len_2 = len(new_stats_2)

    # manually print what we need to convert to a single large string for new_stats_2:
    for idx, stat in enumerate(new_stats_2):
        print(f"  Iteration {idx} stats: {stat}")

    print(f"[ENGINE2-FMHA-OFF] Stats right AFTER Turn 2 => length={after_len_2}")
    turn2_stats_list_e2 = new_stats_2[old_len_2:]
    raw_str_engine2 = convert_stats_to_string(turn2_stats_list_e2)
    print("[ENGINE2-FMHA-OFF] => RAW string of Turn 2 stats:\n", raw_str_engine2)
    parse_res_2 = manual_line_by_line_parser(raw_str_engine2, engine_name="ENGINE2-FMHA-OFF")

    print(f"[ENGINE2-FMHA-OFF] PARSED Turn 2 Stats => TTFT_ms={parse_res_2['TTFT_ms']}, "
          f"prefill={parse_res_2['prefill_iterations']}, decode={parse_res_2['decode_iterations']}")
    finish_reason_2_e2 = "No Output"
    if result2_e2.outputs:
        finish_reason_2_e2 = result2_e2.outputs[0].finish_reason
    print(f"[ENGINE2-FMHA-OFF] Turn 2 finish => {finish_reason_2_e2}")
    print("====================================================\n")

    #----------------------------------------------------------------
    # Comparison
    #----------------------------------------------------------------
    print("==== COMPARISON RESULTS ====")
    print("Engine 1 => WEKA AUGMENTED MEMORY GRID ON:")
    print(f"  TTFT_ms: {parse_res_1['TTFT_ms']}")
    print(f"  prefill_iterations: {parse_res_1['prefill_iterations']}")
    print(f"  decode_iterations:  {parse_res_1['decode_iterations']}")

    print("\nEngine 2 => WEKA AUGMENTED MEMORY GRID OFF:")
    print(f"  TTFT_ms: {parse_res_2['TTFT_ms']}")
    print(f"  prefill_iterations: {parse_res_2['prefill_iterations']}")
    print(f"  decode_iterations:  {parse_res_2['decode_iterations']}")

    # If we want to do some naive % difference:
    ttft1 = parse_res_1['TTFT_ms']
    ttft2 = parse_res_2['TTFT_ms']
    if ttft1 and ttft2 and (ttft1>0) and (ttft2>0):
        ratio = (ttft2 - ttft1)/ttft1 * 100.0
        print(f"\nTTFT difference (Engine2 vs Engine1): {ratio:.2f}%")
    else:
        print("\nCould not compute % difference for TTFT (None or zero).")

    print("==== END COMPARISON RESULTS ====")
    print("\nAll evaluations complete. Exiting.\n")


if __name__ == "__main__":
    main()
