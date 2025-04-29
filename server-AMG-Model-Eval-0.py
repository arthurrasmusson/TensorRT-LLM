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
     - If new stats appended, parse the newly appended slice.
       Otherwise parse the entire new_stats if no new records appended.
  4) Convert them to a raw multi-line string for debugging, parse them
     line-by-line, and compute TTFT.

** Special Debugging Focus **
We've added *extremely* verbose debug prints around stats usage and the manual parser
to ensure we see exactly what data is being handled.
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
        rec_str = json.dumps(rec)  # valid JSON representation
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
      - "Decode" begins once avgNumDecodedTokensPerIter>0.0 or numGenRequests>0
    """
    print(f"\n>>> ENTERING manual_line_by_line_parser() for {engine_name} <<<")
    lines = raw_stats_string.splitlines()

    reconstructed = []
    # We'll parse each line that starts with "Record X:"
    for line_idx, line in enumerate(lines):
        line_str = line.strip()
        print(f"[DEBUG] {engine_name} parser line[{line_idx}]: {line_str}")

        if line_str.startswith("Record "):
            colon_pos = line_str.find(":")
            if colon_pos < 0:
                continue
            dict_part = line_str[colon_pos+1:].strip()
            if dict_part.endswith(","):
                dict_part = dict_part[:-1].strip()
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
# Main Script
###############################################################################
@click.command()
def main():
    engine1_path = "/mnt/weka/Models/Engines/llama-3.1-8b-engine-EVAL-FMHA-ON"
    engine2_path = "/mnt/weka/Models/Engines/llama-3.1-8b-engine-EVAL-FMHA-OFF"
    tokenizer_path = "/mnt/weka/Models/Safetensors/Meta-Llama-3.1-8B-Instruct"

    build_config = BuildConfig(
        max_beam_width=1,
        max_batch_size=8,
        max_num_tokens=2048,
        max_seq_len=8192
    )

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

    # Retention config for ENGINE1, none for ENGINE2
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

    ###########################################################################
    # ENGINE 1 (FMHA ON)
    ###########################################################################
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

    # Turn 2
    turn2_text = book_text + "\n\nwrite me the next chapter"
    turn2_ids = engine1.tokenizer.encode(turn2_text)
    turn2_ids = truncate_tokens(turn2_ids, desc="Turn 2 Prompt")
    print(f"[ENGINE1-FMHA-ON] Turn 2 => {len(turn2_ids)} tokens")

    old_stats_1 = engine1.get_stats()
    old_len_1 = len(old_stats_1)
    print(f"[ENGINE1-FMHA-ON] old_stats_1 length={old_len_1}")

    # Generate Turn 2
    result2 = engine1.generate(
        inputs=turn2_ids,
        kv_cache_retention_config=retention1
    )
    new_stats_1 = engine1.get_stats()
    after_len_1 = len(new_stats_1)
    print(f"[ENGINE1-FMHA-ON] new_stats_1 length={after_len_1}")

    # Decide which slice to parse
    if after_len_1 > old_len_1:
        # parse only appended portion
        turn2_stats_list_1 = new_stats_1[old_len_1:]
        print(f"[ENGINE1-FMHA-ON] Using appended slice of stats => length={len(turn2_stats_list_1)}")
    else:
        # no new stats appended => parse entire new_stats
        turn2_stats_list_1 = new_stats_1
        print(f"[ENGINE1-FMHA-ON] No appended stats => parse entire new_stats => length={len(turn2_stats_list_1)}")

    raw_str_1 = convert_stats_to_string(turn2_stats_list_1)
    print("[ENGINE1-FMHA-ON] => RAW string of Turn 2 stats:\n", raw_str_1)
    parse_res_1 = manual_line_by_line_parser(raw_str_1, engine_name="ENGINE1-FMHA-ON")

    # Print parse results
    print(f"[ENGINE1-FMHA-ON] PARSED Turn 2 Stats => TTFT_ms={parse_res_1['TTFT_ms']}, "
          f"prefill={parse_res_1['prefill_iterations']}, decode={parse_res_1['decode_iterations']}")

    finish_reason_2 = "No Output"
    if result2.outputs and len(result2.outputs) > 0:
        finish_reason_2 = result2.outputs[0].finish_reason
    print(f"[ENGINE1-FMHA-ON] Turn 2 finish => {finish_reason_2}")
    print("====================================================\n")


    ###########################################################################
    # ENGINE 2 (FMHA OFF)
    ###########################################################################
    engine2 = create_llm(
        engine2_path, tokenizer_path,
        policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        block_reuse=False
    )
    print("\n====================================================")
    print("Running ENGINE2-FMHA-OFF (WEKA AUGMENTED MEMORY GRID OFF)")
    print("====================================================\n")

    # Turn 1
    turn1_ids_e2 = engine2.tokenizer.encode(book_text)
    turn1_ids_e2 = truncate_tokens(turn1_ids_e2, desc="Turn 1 Prompt")
    print(f"[ENGINE2-FMHA-OFF] Turn 1 => {len(turn1_ids_e2)} tokens")

    result1_e2 = engine2.generate(
        inputs=turn1_ids_e2,
        kv_cache_retention_config=retention2
    )

    # Turn 2
    turn2_text_e2 = book_text + "\n\nwrite me the next chapter"
    turn2_ids_e2 = engine2.tokenizer.encode(turn2_text_e2)
    turn2_ids_e2 = truncate_tokens(turn2_ids_e2, desc="Turn 2 Prompt")
    print(f"[ENGINE2-FMHA-OFF] Turn 2 => {len(turn2_ids_e2)} tokens")

    old_stats_2 = engine2.get_stats()
    old_len_2 = len(old_stats_2)
    print(f"[ENGINE2-FMHA-OFF] old_stats_2 length={old_len_2}")

    result2_e2 = engine2.generate(
        inputs=turn2_ids_e2,
        kv_cache_retention_config=retention2
    )
    new_stats_2 = engine2.get_stats()
    after_len_2 = len(new_stats_2)
    print(f"[ENGINE2-FMHA-OFF] new_stats_2 length={after_len_2}")

    if after_len_2 > old_len_2:
        turn2_stats_list_2 = new_stats_2[old_len_2:]
        print(f"[ENGINE2-FMHA-OFF] Using appended slice => length={len(turn2_stats_list_2)}")
    else:
        turn2_stats_list_2 = new_stats_2
        print(f"[ENGINE2-FMHA-OFF] No appended stats => parse entire new_stats => length={len(turn2_stats_list_2)}")

    raw_str_2 = convert_stats_to_string(turn2_stats_list_2)
    print("[ENGINE2-FMHA-OFF] => RAW string of Turn 2 stats:\n", raw_str_2)
    parse_res_2 = manual_line_by_line_parser(raw_str_2, engine_name="ENGINE2-FMHA-OFF")

    print(f"[ENGINE2-FMHA-OFF] PARSED Turn 2 Stats => TTFT_ms={parse_res_2['TTFT_ms']}, "
          f"prefill={parse_res_2['prefill_iterations']}, decode={parse_res_2['decode_iterations']}")

    finish_reason_2_e2 = "No Output"
    if result2_e2.outputs and len(result2_e2.outputs) > 0:
        finish_reason_2_e2 = result2_e2.outputs[0].finish_reason
    print(f"[ENGINE2-FMHA-OFF] Turn 2 finish => {finish_reason_2_e2}")
    print("====================================================\n")


    ###########################################################################
    # Comparison
    ###########################################################################
    print("==== COMPARISON RESULTS ====")
    print("Engine 1 => WEKA AUGMENTED MEMORY GRID ON:")
    print(f"  TTFT_ms: {parse_res_1['TTFT_ms']}")
    print(f"  prefill_iterations: {parse_res_1['prefill_iterations']}")
    print(f"  decode_iterations:  {parse_res_1['decode_iterations']}")

    print("\nEngine 2 => WEKA AUGMENTED MEMORY GRID OFF:")
    print(f"  TTFT_ms: {parse_res_2['TTFT_ms']}")
    print(f"  prefill_iterations: {parse_res_2['prefill_iterations']}")
    print(f"  decode_iterations:  {parse_res_2['decode_iterations']}")

    ttft1 = parse_res_1['TTFT_ms']
    ttft2 = parse_res_2['TTFT_ms']
    if ttft1 and ttft2 and (ttft1 > 0) and (ttft2 > 0):
        ratio = (ttft2 - ttft1) / ttft1 * 100.0
        print(f"\nTTFT difference (Engine2 vs Engine1): {ratio:.2f}%")
    else:
        print("\nCould not compute % difference for TTFT (None or zero).")

    print("==== END COMPARISON RESULTS ====")
    print("\nAll evaluations complete. Exiting.\n")


if __name__ == "__main__":
    main()
