import json
from pathlib import Path

def display(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    print(f"\n==========================================")
    print(f"=== ANALYSIS: {path} ===")
    print(f"==========================================")
    for scenario, res in data.items():
        print(f"\nScenario: {scenario} (n={res['n']})")
        print(f"  Numerical pass: {res['numerical_pass_count']}/{res['n']}, Overall accepted: {res['overall_accepted_count']}/{res['n']}")
        print(f"  Routing crossings: {res['routing_crossings']}, Near-threshold baselines: {res['near_threshold_baselines']}")
        metrics = res['metric_order']
        mean = res['mean']
        median = res['median']
        p95 = res['p95']
        max_val = res['max']
        ci95 = res['ci95']
        for i, m in enumerate(metrics):
            print(f"  {m:24s}: mean={mean[i]:+.6f} [{ci95[0][i]:+.6f}, {ci95[1][i]:+.6f}], median={median[i]:+.6f}, p95={p95[i]:+.6f}, max={max_val[i]:+.6f}")
        if res['crossing_mean_L1'] is not None:
            print(f"  crossing_mean_L1: {res['crossing_mean_L1']:.6f}")
        if res['noncrossing_mean_L1'] is not None:
            print(f"  noncrossing_mean_L1: {res['noncrossing_mean_L1']:.6f}")

if __name__ == '__main__':
    display('Saved Model/stage2_stability_repaired_v1/pilot/analysis.json')
    display('Saved Model/stage2_stability_repaired_v1/full/analysis.json')
