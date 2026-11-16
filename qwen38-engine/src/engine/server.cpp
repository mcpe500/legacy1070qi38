#include "server.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "httplib.h"
#include "nlohmann/json.hpp"

extern const unsigned char kWebuiIndex[];
extern const size_t kWebuiIndexSize;

using json = nlohmann::json;

namespace {

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

constexpr const char * kModelName = "qwen3.8-27b";

long long status_kb(const char * key) {
    std::ifstream f("/proc/self/status");
    std::string line;
    const size_t kl = std::strlen(key);
    while (std::getline(f, line)) {
        if (line.compare(0, kl, key) == 0) return std::atoll(line.c_str() + kl);
    }
    return -1;
}

std::string dump(const json & j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }

std::string trim(const std::string & s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

// Length of the longest prefix of s that does not end inside a UTF-8 sequence.
size_t utf8_complete_prefix(const std::string & s) {
    size_t n = s.size();
    if (n == 0) return 0;
    size_t i = n;
    int back = 0;
    while (i > 0 && back < 4 && (static_cast<unsigned char>(s[i - 1]) & 0xC0) == 0x80) { --i; ++back; }
    if (i == 0) return n;               // only continuation bytes: let them through
    const unsigned char lead = static_cast<unsigned char>(s[i - 1]);
    int need = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;
    if (need == 1) return n;
    return (n - (i - 1)) >= static_cast<size_t>(need) ? n : i - 1;
}

struct Msg { std::string role, content, reasoning; bool has_reasoning = false; };

std::string content_text(const json & c) {
    if (c.is_string()) return c.get<std::string>();
    std::string out;
    if (c.is_array()) {
        for (const auto & it : c) {
            if (it.is_object() && it.contains("text") && it["text"].is_string()) out += it["text"].get<std::string>();
            else if (it.is_string()) out += it.get<std::string>();
        }
    }
    return out;
}

// An assistant message from a client may carry the raw streamed text "<think>\nreasoning\n</think>\n\nanswer".
void split_think(Msg & m) {
    if (m.has_reasoning) return;
    const size_t e = m.content.find("</think>");
    if (e == std::string::npos) return;
    std::string r = m.content.substr(0, e);
    const size_t s = r.find("<think>");
    if (s != std::string::npos) r = r.substr(s + 7);
    m.reasoning = r;
    m.has_reasoning = true;
    m.content = m.content.substr(e + 8);
}

struct Piece { std::string text; bool special; };

// Qwen3.8 chat template (the jinja template of the GGUF, text-only, no tools). preserve_thinking is on (template default).
std::vector<Piece> build_pieces(const std::vector<Msg> & msgs, bool thinking, const std::string & effort_in) {
    std::vector<Piece> out;
    size_t first = 0;
    std::string sys;
    while (first < msgs.size() && (msgs[first].role == "system" || msgs[first].role == "developer")) {
        const std::string t = trim(msgs[first].content);
        if (!t.empty()) sys += (sys.empty() ? "" : "\n") + t;
        ++first;
    }
    std::string ri;
    if (thinking) {
        std::string eff = effort_in.empty() ? "xhigh" : effort_in;
        if (eff == "high") eff = "xhigh";
        if (eff == "xhigh")
            ri = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, "
                 "and prioritize correctness, consistency, and clarity in the final answer.";
        else if (eff == "low")
            ri = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";
        else if (eff != "medium") throw std::runtime_error("Unexpected reasoning effort " + eff + ". Supported types are xhigh (default), medium, and low.");
    }
    if (!sys.empty()) {
        out.push_back({"<|im_start|>system\n", true});
        out.push_back({(ri.empty() ? "" : ri + "\n\n") + sys, false});
        out.push_back({"<|im_end|>\n", true});
    } else if (!ri.empty()) {
        out.push_back({"<|im_start|>system\n", true});
        out.push_back({ri, false});
        out.push_back({"<|im_end|>\n", true});
    }
    for (size_t i = first; i < msgs.size(); ++i) {
        Msg m = msgs[i];
        if (m.role == "user") {
            out.push_back({"<|im_start|>user\n", true});
            out.push_back({trim(m.content), false});
            out.push_back({"<|im_end|>\n", true});
        } else if (m.role == "assistant") {
            split_think(m);
            out.push_back({"<|im_start|>assistant\n<think>\n", true});
            out.push_back({trim(m.reasoning), false});
            out.push_back({"\n</think>\n\n", true});
            out.push_back({trim(m.content), false});
            out.push_back({"<|im_end|>\n", true});
        } else {
            throw std::runtime_error("Unexpected message role: " + m.role);
        }
    }
    out.push_back({thinking ? "<|im_start|>assistant\n<think>\n" : "<|im_start|>assistant\n<think>\n\n</think>\n\n", true});
    return out;
}

json error_body(const std::string & msg, const std::string & type) { return json{{"error", {{"message", msg}, {"type", type}}}}; }

}  // namespace

int run_server(Engine & eng, Generator & gen, const EngineConfig & cfg, const ServerOptions & opt) {
    httplib::Server svr;
    std::mutex gen_mtx;
    std::atomic<bool> busy{false};
    std::atomic<long long> n_requests{0};
    const auto t_boot = Clock::now();

    auto stats_json = [&]() {
        return json{{"vram_mib", eng.vram_used_mib()},
                    {"rss_mib", static_cast<double>(status_kb("VmRSS:")) / 1024.0},
                    {"rss_peak_mib", static_cast<double>(status_kb("VmHWM:")) / 1024.0},
                    {"rss_anon_mib", static_cast<double>(status_kb("RssAnon:")) / 1024.0},
                    {"busy", busy.load()},
                    {"cached_tokens", gen.cached_tokens()},
                    {"ctx", cfg.ctx_max},
                    {"gpu_layers", cfg.gpu_layers},
                    {"kv", cfg.kv_iso3 ? "iso3" : "f32"},
                    {"mtp_available", eng.has_mtp()},
                    {"mtp_steps_default", cfg.mtp_steps},
                    {"uptime_s", secs(t_boot, Clock::now())},
                    {"requests", n_requests.load()}};
    };

    auto stats_of = [](const GenStats & st, double live_tg) {
        const int fresh = st.prompt_tokens - st.reused_tokens;
        return json{{"prompt_tokens", st.prompt_tokens}, {"reused_tokens", st.reused_tokens}, {"completion_tokens", st.generated},
                    {"pp_tok_s", st.prefill_s > 0 ? fresh / st.prefill_s : 0.0}, {"prefill_s", st.prefill_s},
                    {"tg_tok_s", live_tg}, {"ttft_ms", st.ttft_s * 1000.0}, {"mtp_steps", st.mtp_steps},
                    {"draft_n", st.draft_n}, {"draft_accepted", st.draft_acc},
                    {"accept_pct", st.draft_n > 0 ? 100.0 * st.draft_acc / st.draft_n : 0.0}};
    };

    svr.Get("/", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(reinterpret_cast<const char *>(kWebuiIndex), kWebuiIndexSize, "text/html; charset=utf-8");
    });
    svr.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content(dump(json{{"status", "ok"}, {"busy", busy.load()}}), "application/json");
    });
    svr.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content(dump(json{{"object", "list"},
                                  {"data", json::array({json{{"id", kModelName}, {"object", "model"}, {"owned_by", "qwen38-engine"},
                                                             {"created", 0}, {"meta", {{"n_ctx", cfg.ctx_max}, {"n_vocab", eng.n_vocab}}}}})}}),
                        "application/json");
    });
    svr.Get("/v1/stats", [&](const httplib::Request &, httplib::Response & res) { res.set_content(dump(stats_json()), "application/json"); });
    svr.Post("/v1/cache/clear", [&](const httplib::Request &, httplib::Response & res) {
        std::lock_guard<std::mutex> lk(gen_mtx);
        gen.reset_cache();
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
        auto fail = [&](int code, const std::string & msg, const char * type = "invalid_request_error") {
            res.status = code;
            res.set_content(dump(error_body(msg, type)), "application/json");
        };
        json body;
        try { body = json::parse(req.body); } catch (const std::exception & e) { fail(400, std::string("invalid JSON: ") + e.what()); return; }
        if (!body.is_object() || !body.contains("messages") || !body["messages"].is_array() || body["messages"].empty()) {
            fail(400, "messages must be a non-empty array");
            return;
        }
        std::vector<Msg> msgs;
        for (const auto & m : body["messages"]) {
            if (!m.is_object() || !m.contains("role") || !m["role"].is_string()) { fail(400, "each message needs a role"); return; }
            Msg x;
            x.role = m["role"].get<std::string>();
            if (m.contains("content") && !m["content"].is_null()) x.content = content_text(m["content"]);
            if (m.contains("reasoning_content") && m["reasoning_content"].is_string()) { x.reasoning = m["reasoning_content"].get<std::string>(); x.has_reasoning = true; }
            msgs.push_back(std::move(x));
        }
        if (msgs.back().role == "assistant") { fail(400, "the last message must be from the user"); return; }

        auto num = [&](const char * k, double d) { return body.contains(k) && body[k].is_number() ? body[k].get<double>() : d; };
        GenParams gp;
        gp.sp.temperature = static_cast<float>(num("temperature", 0.7));
        gp.sp.top_k = static_cast<int>(num("top_k", 20));
        gp.sp.top_p = static_cast<float>(num("top_p", 0.95));
        gp.sp.min_p = static_cast<float>(num("min_p", 0.0));
        gp.sp.repeat_penalty = static_cast<float>(num("repeat_penalty", 1.0));
        gp.sp.presence_penalty = static_cast<float>(num("presence_penalty", 0.0));
        gp.sp.repeat_last_n = static_cast<int>(num("repeat_last_n", 64));
        gp.sp.seed = body.contains("seed") && body["seed"].is_number_integer() ? body["seed"].get<long long>() : -1;
        gp.max_tokens = static_cast<int>(num("max_tokens", num("max_completion_tokens", 512)));
        gp.mtp_steps = static_cast<int>(num("mtp_steps", cfg.mtp_steps));
        gp.reuse = !(body.contains("cache_prompt") && body["cache_prompt"].is_boolean() && !body["cache_prompt"].get<bool>());
        if (gp.sp.temperature < 0 || gp.sp.top_k < 0 || gp.sp.top_p <= 0 || gp.sp.top_p > 1 || gp.sp.min_p < 0 || gp.sp.min_p > 1 ||
            gp.sp.repeat_penalty <= 0 || gp.max_tokens < 1 || gp.mtp_steps < 0 || gp.mtp_steps > 3) {
            fail(400, "a sampler parameter is out of range (temperature >= 0, top_k >= 0, 0 < top_p <= 1, 0 <= min_p <= 1, repeat_penalty > 0, max_tokens >= 1, mtp_steps 0..3)");
            return;
        }
        const bool thinking = !(body.contains("enable_thinking") && body["enable_thinking"].is_boolean() && !body["enable_thinking"].get<bool>());
        const std::string effort = body.contains("reasoning_effort") && body["reasoning_effort"].is_string() ? body["reasoning_effort"].get<std::string>() : "";
        const bool stream = body.contains("stream") && body["stream"].is_boolean() && body["stream"].get<bool>();

        std::vector<int32_t> ids;
        try {
            // Markup pieces are tokenized with special tokens enabled, client text without (it can never become <|im_end|> etc.).
            // Empty client text is dropped and neighbouring markup is merged first, so "<think>\n" + "" + "\n</think>" tokenizes like the
            // text the model generated ("\n\n" is one token), which keeps the cached prefix of the previous turn valid.
            std::vector<Piece> merged;
            for (const Piece & p : build_pieces(msgs, thinking, effort)) {
                if (p.text.empty()) continue;
                if (p.special && !merged.empty() && merged.back().special) merged.back().text += p.text;
                else merged.push_back(p);
            }
            for (const Piece & p : merged) {
                const std::vector<int32_t> t = eng.tok.encode(p.text, p.special);
                ids.insert(ids.end(), t.begin(), t.end());
            }
        } catch (const std::exception & e) { fail(400, e.what()); return; }
        if (static_cast<int>(ids.size()) + 1 >= cfg.ctx_max) {
            fail(400, "the prompt has " + std::to_string(ids.size()) + " tokens, the context is " + std::to_string(cfg.ctx_max), "context_length_exceeded");
            return;
        }
        const std::string id = "chatcmpl-" + std::to_string(++n_requests);
        const long long created = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        // Shared by both modes: runs the generation with the engine lock held and calls `emit_text` with UTF-8-safe pieces.
        if (!stream) {
            std::lock_guard<std::mutex> lk(gen_mtx);
            busy = true;
            GenStats st;
            std::string text, pending;
            try {
                gen.run(ids, gp, [&](int32_t t) {
                    pending += eng.tok.piece(t, true);
                    const size_t n = utf8_complete_prefix(pending);
                    text += pending.substr(0, n);
                    pending.erase(0, n);
                    return true;
                }, st);
            } catch (const std::exception & e) { busy = false; fail(500, e.what(), "server_error"); return; }
            busy = false;
            text += pending;
            std::string full = thinking ? "<think>\n" + text : text;
            json msg = {{"role", "assistant"}};
            const size_t e = full.find("</think>");
            if (thinking && e != std::string::npos) {
                msg["reasoning_content"] = trim(full.substr(8, e - 8));
                msg["content"] = trim(full.substr(e + 8));
            } else {
                msg["content"] = full;
            }
            json x = stats_of(st, st.decode_s > 0 ? st.generated / st.decode_s : 0.0);
            x.update(stats_json());
            res.set_content(dump(json{{"id", id}, {"object", "chat.completion"}, {"created", created}, {"model", kModelName},
                                      {"choices", json::array({json{{"index", 0}, {"message", msg}, {"finish_reason", st.finish == "stop" ? "stop" : "length"}}})},
                                      {"usage", {{"prompt_tokens", st.prompt_tokens}, {"completion_tokens", st.generated}, {"total_tokens", st.prompt_tokens + st.generated}}},
                                      {"x_stats", x}}),
                            "application/json");
            return;
        }

        res.set_header("Cache-Control", "no-cache");
        res.set_header("X-Accel-Buffering", "no");
        res.set_chunked_content_provider("text/event-stream", [&, ids, gp, id, created, thinking](size_t, httplib::DataSink & sink) {
            std::lock_guard<std::mutex> lk(gen_mtx);
            busy = true;
            bool open = true;
            auto send = [&](const json & j) {
                if (!open) return false;
                const std::string s = "data: " + dump(j) + "\n\n";
                if (!sink.write(s.data(), s.size())) open = false;
                return open;
            };
            auto chunk = [&](const json & delta, const json & finish, const json * extra) {
                json j = {{"id", id}, {"object", "chat.completion.chunk"}, {"created", created}, {"model", kModelName},
                          {"choices", json::array({json{{"index", 0}, {"delta", delta}, {"finish_reason", finish}}})}};
                if (extra) j["x_stats"] = *extra;
                return send(j);
            };
            chunk(json{{"role", "assistant"}, {"content", ""}}, nullptr, nullptr);
            if (thinking) chunk(json{{"content", "<think>\n"}}, nullptr, nullptr);   // the template already put it into the prompt
            GenStats st;
            std::string pending;
            const auto t0 = Clock::now();
            auto t_last = t0;
            Clock::time_point t_first = t0;
            bool got_first = false;
            int n_tok = 0;
            try {
                gen.run(ids, gp, [&](int32_t t) {
                    if (!got_first) { got_first = true; t_first = Clock::now(); }
                    ++n_tok;
                    pending += eng.tok.piece(t, true);
                    const size_t n = utf8_complete_prefix(pending);
                    if (n == 0) return open;
                    json extra;
                    const json * ex = nullptr;
                    const auto now = Clock::now();
                    if (secs(t_last, now) > 0.25) {   // live telemetry about four times a second
                        t_last = now;
                        GenStats live;
                        live.prompt_tokens = static_cast<int>(ids.size());
                        live.generated = n_tok;
                        live.ttft_s = secs(t0, t_first);
                        extra = {{"phase", "decode"}, {"completion_tokens", n_tok},
                                 {"tg_tok_s", secs(t_first, now) > 0 ? (n_tok - 1) / secs(t_first, now) : 0.0},
                                 {"ttft_ms", live.ttft_s * 1000.0}};
                        extra.update(json{{"vram_mib", eng.vram_used_mib()}, {"rss_mib", static_cast<double>(status_kb("VmRSS:")) / 1024.0}});
                        ex = &extra;
                    }
                    const bool ok = chunk(json{{"content", pending.substr(0, n)}}, nullptr, ex);
                    pending.erase(0, n);
                    return ok;
                }, st, [&](int done, int total) {
                    json p = {{"phase", "prefill"}, {"prefill_done", done}, {"prefill_total", total}};
                    chunk(json::object(), nullptr, &p);
                });
            } catch (const std::exception & e) {
                send(error_body(e.what(), "server_error"));
            }
            if (!pending.empty()) chunk(json{{"content", pending}}, nullptr, nullptr);
            busy = false;
            json x = stats_of(st, st.decode_s > 0 ? st.generated / st.decode_s : 0.0);
            x.update(stats_json());
            x["phase"] = "done";
            json last = {{"id", id}, {"object", "chat.completion.chunk"}, {"created", created}, {"model", kModelName},
                         {"choices", json::array({json{{"index", 0}, {"delta", json::object()}, {"finish_reason", st.finish == "stop" ? "stop" : "length"}}})},
                         {"usage", {{"prompt_tokens", st.prompt_tokens}, {"completion_tokens", st.generated}, {"total_tokens", st.prompt_tokens + st.generated}}},
                         {"x_stats", x}};
            send(last);
            if (open) sink.write("data: [DONE]\n\n", 14);
            busy = false;
            sink.done();
            return true;
        });
    });

    std::fprintf(stderr, "server: http://%s:%d  (GET /, /health, /v1/models, /v1/stats; POST /v1/chat/completions)\n", opt.host.c_str(), opt.port);
    if (!svr.listen(opt.host.c_str(), opt.port)) {
        std::fprintf(stderr, "server: cannot listen on %s:%d\n", opt.host.c_str(), opt.port);
        return 1;
    }
    return 0;
}
