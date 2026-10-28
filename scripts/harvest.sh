#!/usr/bin/env bash
# Unattended harvest: find, judge, and label up to N new clips, extract them, open a PR.
#
#   make harvest          # up to 5 clips
#   make harvest N=3
#
# Steps: fresh branch from main -> Claude (headless, docs/HARVEST.md, limited tools) judges
# and labels -> pose extraction + tracking on the Mac GPU -> commit the manifest,
# candidate queue and label log -> push -> PR. Claude's labels stay out of training
# until reviewed in the labeler's Review tab. Logs: runs/logs/harvest_<stamp>.log
set -euo pipefail
N="${1:-5}"
cd "$(dirname "$0")/.."
STAMP="$(date +%Y%m%d-%H%M)"
LOG="runs/logs/harvest_${STAMP}.log"
mkdir -p runs/logs

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Uncommitted changes in tracked files; commit or stash them first." >&2
  exit 1
fi
git checkout -q main
git pull -q --ff-only
BRANCH="harvest-${STAMP}"
git checkout -q -b "$BRANCH"

echo "== harvest $STAMP: up to $N clips, log $LOG"
caffeinate -dims claude -p \
  "Read docs/HARVEST.md and follow it exactly. Accept and label up to ${N} new clips (rescue or normal), then stop and report." \
  --allowedTools \
    "Bash(uv run hydroknight harvest *)" \
    "Bash(uv run hydroknight sheets *)" \
    "Bash(uv run hydroknight write-label *)" \
    "Bash(ffprobe *)" \
    "Read" \
    "Write(scratch/specs/*)" \
  2>&1 | tee "$LOG"

echo "== extraction (Mac GPU)" | tee -a "$LOG"
caffeinate -dims uv run hydroknight harvest extract 2>&1 | tee -a "$LOG"

git add data/manifests/pool_footage.jsonl data/harvest/candidates.jsonl \
  data/annotations/labeled_by_claude.json data/blocklist.txt 2>/dev/null || true
if git diff --cached --quiet; then
  echo "Nothing new to commit." | tee -a "$LOG"
  git checkout -q main && git branch -q -D "$BRANCH"
  exit 0
fi
SUMMARY="$(uv run hydroknight harvest status)"
git commit -q -m "Harvest ${STAMP}: new clips and Claude labels (pending review)

${SUMMARY}

"
git push -q -u origin "$BRANCH"
gh pr create --base main --head "$BRANCH" \
  --title "Harvest ${STAMP}: new clips (labels pending review)" \
  --body "Unattended harvest run. Review every new label in the labeler's Review tab before merging; unreviewed Claude labels are held out of training.

\`\`\`
${SUMMARY}
\`\`\`

Run log (local): ${LOG}

🤖 Generated with [Claude Code](https://claude.com/claude-code)" | tee -a "$LOG"
