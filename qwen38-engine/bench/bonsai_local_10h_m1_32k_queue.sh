#!/usr/bin/env bash
# M1 long-prompt queue of issue #55: 32K-token suites (coding, repetitive), one prefill per suite and head (--state-save),
# then every decode variant from --state-load. Run with ENGINE_BIN set to a frozen build.
set -u
cd "$(dirname "$0")/.."
python3 bench/bonsai_suite_state.py 32768 coding,repetitive none m0
python3 bench/bonsai_suite_state.py 32768 coding,repetitive q4km m0 m2 ad pldad
echo QUEUE_DONE
