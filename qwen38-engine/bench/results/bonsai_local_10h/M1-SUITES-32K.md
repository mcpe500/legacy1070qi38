### Prompt 32768 tokens (32759 rendered; one prefill per suite and head with --state-save, decode variants from --state-load; build with the M2 v3 decode kernel)

| suite | head | variant | prompt | gen | decode tok/s | vs none m0 | accept | pld hits | ids == none m0 | peak VRAM MiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| coding | none | m0 | 32768 | 256 | 10.78 | 1.000x | 0/0 | 0 | identical (bb704398) | 6357 |
| coding | q4km | m0 | 32768 | 256 | 10.76 | 0.998x | 0/0 | 0 | identical (bb704398) | 6829 |
| coding | q4km | m2 | 32768 | 256 | 12.24 | 1.135x | 154/204 | 0 | identical (bb704398) | 6831 |
| coding | q4km | ad | 32768 | 256 | 13.13 | 1.217x | 162/186 | 0 | identical (bb704398) | 6877 |
| coding | q4km | pldad | 32768 | 256 | 12.33 | 1.144x | 156/210 | 10 | identical (bb704398) | 6877 |
| repetitive | none | m0 | 32768 | 256 | 10.74 | 1.000x | 0/0 | 0 | identical (926abe5d) | 6357 |
| repetitive | q4km | m0 | 32768 | 256 | 10.76 | 1.002x | 0/0 | 0 | identical (926abe5d) | 6829 |
| repetitive | q4km | m2 | 32768 | 256 | 14.36 | 1.337x | 170/174 | 0 | identical (926abe5d) | 6831 |
| repetitive | q4km | ad | 32768 | 256 | 15.42 | 1.435x | 197/200 | 0 | identical (926abe5d) | 6877 |
| repetitive | q4km | pldad | 32768 | 256 | 14.13 | 1.315x | 192/225 | 6 | identical (926abe5d) | 6877 |

| suite | head | prefill s | prefill tok/s | state MiB |
|---|---|---:|---:|---:|
| coding | none | 599.2 | 54.67 | 549.5 |
| coding | q4km | 646.9 | 50.64 | 574.5 |
| repetitive | none | 605.1 | 54.14 | 549.5 |
| repetitive | q4km | 646.3 | 50.69 | 574.5 |
