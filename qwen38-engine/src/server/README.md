# src/server

Server HTTP/SSE tertanam dengan cpp-httplib. `POST /v1/chat/completions` menerima format
OpenAI-compatible; tersedia streaming SSE, telemetry, cache prefix, stop string, MTP adaptif, dan
PLD.

Status: **implementasi lengkap dan aktif**. Server memproses satu permintaan pada satu waktu.
Endpoint rinci ada di README utama `qwen38-engine/`.

Baca dulu `.agents/skills/webui-and-parity-bench/SKILL.md`.
