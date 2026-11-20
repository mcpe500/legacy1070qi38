#!/usr/bin/env bash
# M1 queue of issue #55 (run in the background): 512- and 4K-token suites, one server per draft head.
set -u
cd "$(dirname "$0")/.."
S=bench/bonsai_mtp_suite.py
python3 $S run 512 none m0
for h in donor q4km q8; do python3 $S run 512 $h m0 m1 m2 m3 ad pld2 pldad; done
python3 $S run 4096 none m0
python3 $S run 4096 q4km m2 ad pldad
python3 $S run 4096 donor m2 ad
python3 $S run 4096 q8 m2
echo QUEUE_DONE
