# Next Session — Execute Leps Localizer Training Plan

## State at hand-off (2026-05-05)

- Branch `feat/leps-localizer-training` pushed to `RolnickLab/ami-ml` (5 commits).
- Spec: `research/leps_localizer/DESIGN.md` (approved).
- Plan: `docs/superpowers/plans/2026-05-05-leps-localizer-training.md` (18 tasks, 5 phases).
- VM `ami-workspace-02-gpu` already provisioned (H100 24GB) with FUSE-mount tooling installed.
- Eval set locked: 4 COCO datasets at `~/Projects/Fieldguide/chroma-backend/.claude/worktrees/detector-training/detector_dataset/datasets/`.

## First moves

1. **Pull plan into context**: read `docs/superpowers/plans/2026-05-05-leps-localizer-training.md` end-to-end.
2. **Decide execution mode** (skill suggests subagent-driven; for this multi-VM, multi-repo work the inline-plus-handoff hybrid works fine):
   - Phase 1 (extract pipeline) → execute on laptop or beast.
   - Phase 2 (VM bootstrap) → execute via SSH to `ami-workspace-02-gpu`.
   - Phase 3 (ami-ml laptop side) → execute on laptop, push.
   - Phase 4 (training) → SSH to VM, launch `nohup`, monitor via wandb.
   - Phase 5 (RT-DETR + FRCNN + comparison) → same as 4.
3. **Phase 1 Task 1 first**: implement `IndexRecord` JSONL writer in `detector_dataset/src/detector_dataset/index_jsonl.py`. TDD style — test then code.

## Working directories

- Plan + spec: `~/Projects/AMI/ami-ml` (branch `feat/leps-localizer-training`)
- Extract pipeline: `~/Projects/Fieldguide/chroma-backend/.claude/worktrees/detector-training/detector_dataset/` (branch `worktree-detector-training`)
- Infra unit file: `~/Projects/AMI/ami-devops/systemd/storage/`

## Connect to VM

```
ssh ami-workspace-02-gpu          # 192.168.129.83 via ami-arbutus-bastion
# key: ~/.ssh/ami2026.pem (loaded by ssh config)
# CAUTION: agent-forwarded git writes commit author as adityajain07.
# Override per-session:
#   GIT_AUTHOR_NAME='Michael Bunsen' GIT_AUTHOR_EMAIL='michael@mixedneeds.com' \
#   GIT_COMMITTER_NAME='Michael Bunsen' GIT_COMMITTER_EMAIL='michael@mixedneeds.com' \
#     <command>
```

## Open decisions for next session

- YOLOv11 vs YOLOv8 → default v11
- `imgsz` 640 vs 1024 → run 640 first, 1024 second
- Square-target augmentation → skip first run
- `global_butterflies_2604` 12TB squashfs already on box → confirm provenance before using
- Disk strategy → start with FUSE streaming, switch to bulk copy if dataloader I/O bottlenecks

## Production safety (still relevant)

- FG DB queries are read-only with `SET LOCAL statement_timeout`
- FG image fetches go through Arbutus S3, not the FG app server
- Production EC2 sees zero traffic from training pipeline

## References

- Spec: `research/leps_localizer/DESIGN.md`
- Plan: `docs/superpowers/plans/2026-05-05-leps-localizer-training.md`
- VM setup history: `~/Projects/AMI/ami-devops/docs/claude/sessions/2026-04-28-object-store-fuse-mount-setup.md`
- Old-cloud GPU runbook: `~/Projects/AMI/ami-devops/docs/claude/sessions/2026-04-20-ami-gpu-04-provisioning.md`
- Memory entry: `~/.claude/projects/-home-michael-Projects-Fieldguide-chroma-backend/memory/leps-localizer-training.md`
