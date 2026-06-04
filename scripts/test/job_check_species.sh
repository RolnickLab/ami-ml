#!/bin/bash
#SBATCH --job-name=check_species
#SBATCH --account=def-drolnick
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=/project/6068129/melabbas/data/ne-america-eccv2024/check_species-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hack1996man@gmail.com

set -euo pipefail

module load python/3.11

python3 - <<'EOF'
import json, urllib.request, concurrent.futures, time

with open('/project/6068129/melabbas/data/ne-america-eccv2024/metadata/01_ami-gbif_fine-grained_ne-america_category_map.json') as f:
    cat_map = json.load(f)

keys = list(cat_map.keys())
print(f'Total species: {len(keys)}', flush=True)

butterfly_families = {'Papilionidae','Pieridae','Nymphalidae','Lycaenidae','Riodinidae','Hesperiidae'}
results = {}

def fetch(k):
    for attempt in range(3):
        try:
            url = f'https://api.gbif.org/v1/species/{k}'
            with urllib.request.urlopen(url, timeout=10) as r:
                d = json.load(r)
                return k, d.get('canonicalName','?'), d.get('family','unknown'), d.get('order','unknown')
        except Exception as e:
            time.sleep(1)
    return k, '?', 'error', 'error'

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(fetch, k): k for k in keys}
    done = 0
    for f in concurrent.futures.as_completed(futures):
        k, name, fam, order = f.result()
        results[k] = {'name': name, 'family': fam, 'order': order}
        done += 1
        if done % 100 == 0:
            print(f'  {done}/{len(keys)}...', flush=True)

# Save full results
out_path = '/project/6068129/melabbas/data/ne-america-eccv2024/metadata/species_lookup.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved to {out_path}', flush=True)

# Summary
families = {}
for v in results.values():
    fam = v['family']
    families[fam] = families.get(fam, 0) + 1

print('\nTop 20 families:')
for fam, cnt in sorted(families.items(), key=lambda x: -x[1])[:20]:
    tag = ' <- BUTTERFLY' if fam in butterfly_families else ''
    print(f'  {fam}: {cnt}{tag}')

butterfly_count = sum(v for k,v in families.items() if k in butterfly_families)
other_count = sum(v for k,v in families.items() if k not in butterfly_families and k not in ('unknown','error'))
print(f'\nButterflies: {butterfly_count}')
print(f'Moths (approx): {other_count}')
EOF

~/bin/notify "check_species done" "Species family lookup complete. Results in data/ne-america-eccv2024/metadata/species_lookup.json"
