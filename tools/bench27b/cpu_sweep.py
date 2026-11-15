#!/usr/bin/env python3
"""Sapuan thread llama-bench (tg saja) sambil mencatat pemakaian CPU.
Pemakaian: cpu_sweep.py <bin> <model> <ngl> <out.json> t1 t2 ..."""
import json, subprocess, sys, time, threading

binp, model, ngl, out = sys.argv[1:5]
threads = [int(t) for t in sys.argv[5:]]
CLK = 100  # os.sysconf('SC_CLK_TCK')


def cpu_times():
    per = []
    for line in open('/proc/stat'):
        if line.startswith('cpu') and line[3] != ' ':
            v = [int(x) for x in line.split()[1:]]
            idle = v[3] + v[4]
            per.append((sum(v), idle))
    return per


def proc_cpu(pid):
    try:
        f = open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()
        return (int(f[11]) + int(f[12])) / CLK
    except FileNotFoundError:
        return None


results = []
for t in threads:
    args = [f'{binp}/llama-bench', '-m', model, '-ngl', ngl, '-p', '0', '-n', '64', '-r', '3', '-t', str(t), '-o', 'json']
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    samples = []
    stop = False

    def sampler():
        prev = cpu_times(); prev_p = proc_cpu(p.pid); prev_t = time.time()
        while not stop:
            time.sleep(1)
            cur = cpu_times(); cur_p = proc_cpu(p.pid); now = time.time()
            if cur_p is None or prev_p is None:
                break
            busy = [1 - (ci - pi) / max(1, (ct - pt)) for (ct, ci), (pt, pi) in zip(cur, prev)]
            samples.append({'proc_cores': (cur_p - prev_p) / (now - prev_t), 'per_cpu_busy': busy})
            prev, prev_p, prev_t = cur, cur_p, now

    th = threading.Thread(target=sampler); th.start()
    stdout, _ = p.communicate()
    stop = True; th.join()
    d = json.loads(stdout)
    # buang 5 detik pertama (pemuatan model)
    s = samples[5:] or samples
    proc = sorted(x['proc_cores'] for x in s)
    busy = [sum(x['per_cpu_busy']) / len(x['per_cpu_busy']) for x in s]
    r = {'threads': t, 'tg_tps': d[0]['avg_ts'], 'tg_stddev': d[0]['stddev_ts'],
         'proc_cores_median': proc[len(proc) // 2], 'proc_cores_p90': proc[int(len(proc) * 0.9)],
         'system_busy_pct_mean': 100 * sum(busy) / len(busy),
         'samples': len(s)}
    print(json.dumps(r), flush=True)
    results.append(r)
json.dump(results, open(out, 'w'), indent=1)
