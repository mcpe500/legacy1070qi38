// Embedded HTTP/SSE server (cpp-httplib): OpenAI-compatible POST /v1/chat/completions (stream and non-stream), GET /v1/models,
// GET /health, GET /v1/stats, and GET / with the single-file WebUI compiled into the binary. One generation at a time.
#pragma once
#include <string>

#include "engine.hpp"
#include "generator.hpp"

struct ServerOptions {
    std::string host = "127.0.0.1";   // loopback by default; --host 0.0.0.0 exposes it on the network (no authentication)
    int port = 8080;
    // --state-save with --server: POST /v1/state/save writes the current model state there (the path is fixed at startup,
    // clients cannot choose it), and so does a clean shutdown (SIGINT / SIGTERM).
    std::string state_save;
    std::string model_name;   // general.name of the GGUF, reported by /v1/stats (the API model id stays qwen3.8-27b)
    std::string model_file;   // basename of the GGUF (and of the --mtp-gguf sidecar), reported by /v1/stats
};

int run_server(Engine & eng, Generator & gen, const EngineConfig & cfg, const ServerOptions & opt);
