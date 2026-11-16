// Embedded HTTP/SSE server (cpp-httplib): OpenAI-compatible POST /v1/chat/completions (stream and non-stream), GET /v1/models,
// GET /health, GET /v1/stats, and GET / with the single-file WebUI compiled into the binary. One generation at a time.
#pragma once
#include <string>

#include "engine.hpp"
#include "generator.hpp"

struct ServerOptions {
    std::string host = "127.0.0.1";   // loopback by default; --host 0.0.0.0 exposes it on the network (no authentication)
    int port = 8080;
};

int run_server(Engine & eng, Generator & gen, const EngineConfig & cfg, const ServerOptions & opt);
