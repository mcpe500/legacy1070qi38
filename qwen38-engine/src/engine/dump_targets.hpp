// --dump-targets: the teacher-free T4 precompute pass of the Bonsai MTP/df3 pipeline, native PTQ1_0.
//
// The whole model sits resident (all 64 layers + the LM head on the GPU when the budget allows — ~5.54 GiB of
// PTQ1_0 weights on a T4 15 GiB, no BF16 conversion, no layer-by-layer CPU<->GPU streaming). Per sample:
// `reset()` zeroes the DeltaNet/conv state, one full prefill of seq_len tokens produces the output-normed
// hidden of every position (forward_ex h_rows), then the head gives the full logits of 8 rows at a time and
// the top-K ids/vals + logsumexp are reduced on the CPU. Output: the same package the live
// precompute_teacher_targets.py writes — Python .npz shards OR plain .npy shard directories (loader accepts
// both) — plus dims.json and an atomic manifest.json with per-shard auto-resume.
//
// Everything is written with plain stdio from host buffers allocated once at setup; no allocation inside the
// sample loop beyond the per-shard directory rename.
#pragma once
#include <string>

#include "engine.hpp"

struct DumpConfig {
    std::string tokens_bin;        // int32 little-endian token stream (tokenize_for_engine.py)
    std::string plan_json;         // default: <tokens_bin dir>/plan.json
    std::string targets_out;       // package dir (shards + manifest.json + dims.json)
    int samples = 0;               // 0 = every plan entry
    int seq_len = 0;               // 0 = plan.json seq_len
    int top_k = 64;
    std::string hidden_dtype = "int4g64";  // int4g64 | int8 (std::string: a const char* here held a dangling
                                           // next().c_str() from main.cpp and corrupted the dtype check)
    int shard_samples = 64;
    double max_gb = 0;             // 0 = unlimited; stop with a consistent manifest when the package hits it
    bool dump_f32_hidden = false;  // extra hf32.npy per shard (smoke parity checks; 20 KB/token)
};

// Runs the dump pass to completion; returns the process exit code and prints one JSON summary line.
// Takes the parsed EngineConfig by value: dump mode overrides ctx_max/t_max/gpu_layers/gpu_head/mtp_steps.
int run_dump(EngineConfig cfg, const DumpConfig & dcfg);
