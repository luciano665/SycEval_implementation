import json, glob

def get_mode_rates(filepath, min_idx=0):
    with open(filepath) as f:
        d = json.load(f)
    recs = d.get("individual_records") or d.get("records", [])
    modes = {}
    for r in recs:
        if r.get("idx", 0) < min_idx: continue
        mode = r.get("where") or r.get("mode", "")
        syco = r.get("sycophancy", "")
        if not mode: continue
        modes.setdefault(mode, {"total":0, "regr":0, "prog":0})
        modes[mode]["total"] += 1
        if syco == "regressive": modes[mode]["regr"] += 1
        if syco == "progressive": modes[mode]["prog"] += 1
    def pct(a,b): return a/b*100 if b>0 else 0
    return {m: {
        "overall": pct(v["regr"]+v["prog"], v["total"]),
        "regr": pct(v["regr"], v["total"]),
        "prog": pct(v["prog"], v["total"]),
        "n": v["total"]
    } for m,v in modes.items() if v["total"]>0}

models = ["gemma_1b","gemma_4b","llama_1b","llama_3b","phi_1.5","phi_2"]

for dataset in ["medDataset","healthsearch"]:
    print(f"\n=== {dataset} ===")
    ic_nocp, ic_cp, pre_nocp, pre_cp = [], [], [], []
    ic_nocp_legacy, pre_nocp_legacy = [], []
    for m in models:
        bf = glob.glob(f"results/{dataset}_v6_1000/run_baseline_{m}*_v5.json")
        cf = glob.glob(f"results/{dataset}_v6_1000/run_conformal_{m}*_v6.json")
        if not bf or not cf: continue
        with open(cf[0]) as f:
            calib_conf = json.load(f)
        calibration_items = calib_conf.get("metadata", {}).get("config", {}).get("calibration_items")
        if calibration_items is None:
            print(f"  WARNING: {cf[0]} has no metadata.config.calibration_items; assuming 0 (unpaired baseline)")
            calibration_items = 0
        bm = get_mode_rates(bf[0], min_idx=calibration_items)  # paired: aligns with conformal's test-only idx range
        bm_legacy = get_mode_rates(bf[0])  # legacy: full 1000-item baseline, unpaired with conformal's 750-item test split
        cm = get_mode_rates(cf[0])
        bic  = bm.get("in-context", {})
        bpre = bm.get("preemptive", {})
        bic_legacy  = bm_legacy.get("in-context", {})
        bpre_legacy = bm_legacy.get("preemptive", {})
        cic  = cm.get("in-context", {})
        cpre = cm.get("preemptive", {})
        ic_nocp.append(bic.get("overall",0)); ic_cp.append(cic.get("overall",0))
        pre_nocp.append(bpre.get("overall",0)); pre_cp.append(cpre.get("overall",0))
        ic_nocp_legacy.append(bic_legacy.get("overall",0)); pre_nocp_legacy.append(bpre_legacy.get("overall",0))
        print(f"  {m}: (baseline N: paired IC={bic.get('n',0)} Pre={bpre.get('n',0)} | legacy IC={bic_legacy.get('n',0)} Pre={bpre_legacy.get('n',0)})")
        print(f"    IC:  NoCP={bic.get('overall',0):.1f}% (r={bic.get('regr',0):.1f}% p={bic.get('prog',0):.1f}%)  CP={cic.get('overall',0):.1f}% (r={cic.get('regr',0):.1f}% p={cic.get('prog',0):.1f}%)  Δ={cic.get('overall',0)-bic.get('overall',0):+.1f}%")
        print(f"    Pre: NoCP={bpre.get('overall',0):.1f}% (r={bpre.get('regr',0):.1f}% p={bpre.get('prog',0):.1f}%)  CP={cpre.get('overall',0):.1f}% (r={cpre.get('regr',0):.1f}% p={cpre.get('prog',0):.1f}%)  Δ={cpre.get('overall',0)-bpre.get('overall',0):+.1f}%")
    print(f"  AVG IC:  {sum(ic_nocp)/len(ic_nocp):.1f}% -> {sum(ic_cp)/len(ic_cp):.1f}% ({sum(ic_cp)/len(ic_cp)-sum(ic_nocp)/len(ic_nocp):+.1f}%)")
    print(f"  AVG Pre: {sum(pre_nocp)/len(pre_nocp):.1f}% -> {sum(pre_cp)/len(pre_cp):.1f}% ({sum(pre_cp)/len(pre_cp)-sum(pre_nocp)/len(pre_nocp):+.1f}%)")
    print(f"  [legacy full-1000 baseline, unpaired] AVG IC NoCP:  {sum(ic_nocp_legacy)/len(ic_nocp_legacy):.1f}%  (paired={sum(ic_nocp)/len(ic_nocp):.1f}%, Δ={sum(ic_nocp)/len(ic_nocp)-sum(ic_nocp_legacy)/len(ic_nocp_legacy):+.1f}%)")
    print(f"  [legacy full-1000 baseline, unpaired] AVG Pre NoCP: {sum(pre_nocp_legacy)/len(pre_nocp_legacy):.1f}%  (paired={sum(pre_nocp)/len(pre_nocp):.1f}%, Δ={sum(pre_nocp)/len(pre_nocp)-sum(pre_nocp_legacy)/len(pre_nocp_legacy):+.1f}%)")
