
import json
import os
import glob
import numpy as np

def calculate_stats(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Determine format
        if isinstance(data, dict) and 'statistical_results' in data:
            # Conformal Format (v2)
            stats = data['statistical_results']['overall_rates']
            return {
                'n': stats['N'],
                'rate': stats['overall'],
                'type': 'Conformal'
            }
        elif isinstance(data, list):
            # Baseline Format (List of records)
            records = data
            n = len(records)
            if n == 0: return None
            
            # Count regressive sycophancy
            # In baseline, we look for 'sycophancy' == 'regressive'
            # OR classifiy based on labels if field missing
            regressive = sum(1 for r in records if r.get('sycophancy') == 'regressive')
            
            return {
                'n': n,
                'rate': regressive / n,
                'type': 'Baseline'
            }
        elif isinstance(data, dict) and 'individual_records' in data:
             # Conformal w/o stats saved (rare)
             records = data['individual_records']
             n = len(records)
             regressive = sum(1 for r in records if r.get('sycophancy') == 'regressive')
             return {'n': n, 'rate': regressive/n, 'type': 'Conformal (Calc)'}
             
        return None
    except Exception as e:
        return {'error': str(e)}

files = sorted(glob.glob('results/final_run_v1/*.json') + glob.glob('results/neutral_suite_v1/*.json'))
results = []

print(f"{'Model':<30} | {'Type':<15} | {'N':<6} | {'Sycophancy %':<12}")
print("-" * 75)

for file in files:
    name = os.path.basename(file)
    if 'thresholds' in name: continue
    
    # Extract Model Name
    model_name = name.replace('_conformal.json', '').replace('_baseline.json', '').replace('_', ' ').title()
    
    stats = calculate_stats(file)
    if stats and 'error' not in stats:
        print(f"{model_name:<30} | {stats['type']:<15} | {stats['n']:<6} | {stats['rate']:.1%}")
    elif stats:
        print(f"{model_name:<30} | ERROR: {stats['error']}")
