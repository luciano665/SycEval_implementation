import sys
import json
import os
import glob

def calculate_stats(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Determine format (Handle nested baselines or flat Conformal)
        recs = data.get('individual_records', []) if isinstance(data, dict) else []
        if not recs and isinstance(data, dict):
             # Maybe Baseline nested?
             for v in data.values():
                if isinstance(v, list) and v and ('sycophancy' in v[0] or 'mode' in v[0]):
                    recs = v; break
        if not recs and isinstance(data, list): recs = data
            
        if not recs: return None
        
        n = len(recs)
        
        # Calculate Rates
        regr = sum(1 for r in recs if r.get('sycophancy') == 'regressive')
        prog = sum(1 for r in recs if r.get('sycophancy') == 'progressive')
        over = regr + prog
        
        # Context Split - Check 'mode', 'where', or 'type'
        ic_recs = [r for r in recs if any(x in str(r.get(f,'')).lower() for f in ['mode','where','type'] for x in ['in-context','ic'])]
        pm_recs = [r for r in recs if any(x in str(r.get(f,'')).lower() for f in ['mode','where','type'] for x in ['preemptive','pm'])]
        
        ic_rate = sum(1 for r in ic_recs if r.get('sycophancy') in ['regressive','progressive'])/len(ic_recs) if ic_recs else 0
        pm_rate = sum(1 for r in pm_recs if r.get('sycophancy') in ['regressive','progressive'])/len(pm_recs) if pm_recs else 0

        return {
            'n': n,
            'over': over/n,
            'regr': regr/n,
            'prog': prog/n,
            'ic': ic_rate,
            'pm': pm_rate
        }
    except Exception as e:
        return {'error': str(e)}

# Get files from args or default globs
if len(sys.argv) > 1:
    files = sys.argv[1:]
else:
    files = sorted(glob.glob('results/final_experiment_v1/*.json') +
                  glob.glob('results/final_run_v1/*.json') + 
                  glob.glob('results/neutral_suite_v1/*.json') +
                  glob.glob('results/ministral_judge_v1/*.json'))

print(f"{'Model':<30} | {'Over':<6} | {'Regr':<6} | {'Prog':<6} | {'IC':<6} | {'PM':<6}")
print("-" * 80)

for file in files:
    name = os.path.basename(file).replace('_conformal.json','').replace('_baseline.json','').replace('_',' ').title()
    if 'Thresholds' in name: continue
    
    stats = calculate_stats(file)
    if stats and 'error' not in stats:
        print(f"{name:<30} | {stats['over']:<6.1%} | {stats['regr']:<6.1%} | {stats['prog']:<6.1%} | {stats['ic']:<6.1%} | {stats['pm']:<6.1%}")
    elif stats:
        print(f"{name:<30} | ERROR: {stats['error']}")
