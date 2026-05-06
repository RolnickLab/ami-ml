#!/bin/bash
#SBATCH --job-name=count_butterfly_imgs
#SBATCH --account=def-drolnick
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=/project/6068129/melabbas/data/count_butterfly_images-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hack1996man@gmail.com

module load python/3.11

python3 - <<'EOF'
import tarfile, glob

shards = sorted(glob.glob('/project/rrg-bengioy-ad/melabbas/northamerica_butterflies_v2/webdataset/**/*.tar', recursive=True))
print(f'Total shards: {len(shards)}', flush=True)

totals = {'train': 0, 'val': 0, 'test': 0}
for shard in shards:
    split = 'train' if '/train/' in shard else 'val' if '/val/' in shard else 'test'
    try:
        with tarfile.open(shard) as tf:
            count = sum(1 for m in tf.getmembers() if m.name.endswith('.jpg'))
            totals[split] += count
    except Exception as e:
        print(f'Error {shard}: {e}', flush=True)

for split, count in totals.items():
    print(f'{split}: {count:,}')
print(f'Total: {sum(totals.values()):,}')
EOF

~/bin/notify "count_butterfly_imgs done" "Image count complete. Results in data/count_butterfly_images-${SLURM_JOB_ID}.out"
