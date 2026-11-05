#!/usr/bin/env bash
# fastconformer-screen.sh — #267's screen steps on cuda-dev.
#
# On the Mac, from the worktree root, one subcommand per step (the repo is staged as a clean
# bundle clone on the box, so every step runs a committed SHA):
#
#   scripts/fastconformer-screen.sh env          # the isolated CPU NeMo venv (never hifzguide's)
#   scripts/fastconformer-screen.sh fetch        # the pinned .nemo, hash checked locally
#   scripts/fastconformer-screen.sh stage-ship-a # every clip Ship A's decode plan reads
#   scripts/fastconformer-screen.sh stage-edits  # every clip the edit render reads (the
#                                                # source/donors of the candidates it tries,
#                                                # not only the kept pairs'), then the render
#
# Both staging steps fill one directory, $ROOT/audio/clips: the decode reads that for Ship A
# and the edit render reads it for its sources, so neither depends on a clip left there by an
# earlier run.
#   scripts/fastconformer-screen.sh decode       # Ship A, the edits and qdat, CPU, gated
#   scripts/fastconformer-screen.sh check [label]
#   scripts/fastconformer-screen.sh fetchout DIR # pull decodes, identities and timings back
#
# Every step checks the 10 GB disk reserve first and runs detached under setsid with stdin
# closed: one log, one pid and one exit-code file under $ROOT/logs. The model runs only in
# $ROOT/venv (created from the box's Python 3.11, never the hifzguide env's site-packages);
# the repo's data tools (staging, rendering) run in hifzguide, which never imports the model.
set -euo pipefail

HOST=${HOST:-root@cuda-dev}
ROOT=${ISSUE_ROOT:-/root/scratch/fc-screen}
REPO=$ROOT/repo
LOGS=$ROOT/logs
WORK=$ROOT/data
AUDIO=$ROOT/audio
CLIPS=$ROOT/audio/clips
EDITS=$ROOT/edits
VENV=$ROOT/venv
CONDA=${CONDA:-/root/miniconda3/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-hifzguide}
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUT=tools/tadabur/fastconformer_screen
PAUSE=${PAUSE:-/root/scratch/tlog-mining/PAUSE}
DISK_RESERVE_GB=${DISK_RESERVE_GB:-10}

step=${1:-}
action=${2:-}
case "$step" in
  env|fetch|stage-ship-a|stage-edits|decode) ;;
  *) echo "usage: $0 {env|fetch|stage-ship-a|stage-edits|decode} [launch|print]" >&2; exit 2 ;;
esac
LABEL=$step

remote() { ssh -n "$HOST" "$@"; }

require_disk() {
  local free
  free=$(remote "df -BG --output=avail /root/scratch | tail -1 | tr -dc 0-9")
  if [ "$free" -lt "$DISK_RESERVE_GB" ]; then
    echo "free disk ${free} GB is below the ${DISK_RESERVE_GB} GB reserve; refusing" >&2
    exit 1
  fi
}

stage_repo() {
  local branch sha
  branch=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)
  sha=$(git -C "$REPO_DIR" rev-parse HEAD)
  remote "mkdir -p '$ROOT' '$LOGS' '$WORK' '$AUDIO' && rm -rf '$REPO' '$ROOT/repo.bundle'"
  git -C "$REPO_DIR" bundle create - "$branch" | ssh "$HOST" "cat > '$ROOT/repo.bundle'"
  remote "git clone -q -b '$branch' '$ROOT/repo.bundle' '$REPO'"
  remote "rm -f '$ROOT/repo.bundle' && test \"\$(git -C '$REPO' rev-parse HEAD)\" = '$sha'" \
    || { echo "the box's clone is not $sha" >&2; return 1; }
}

command_for() {
  case "$step" in
    env) cat <<ENV
test -x '$VENV/bin/python' || /root/miniconda3/envs/hifzguide/bin/python -m venv '$VENV'
'$VENV/bin/pip' install -q -U pip
'$VENV/bin/pip' install -q --index-url https://download.pytorch.org/whl/cpu torch torchaudio
'$VENV/bin/pip' install -q numpy==1.26.4 pyarrow==21.0.0 datasets==4.1.1
'$VENV/bin/pip' install -q wget antlr4-python3-runtime==4.9.3
'$VENV/bin/pip' install -q 'nemo_toolkit[asr]==2.0.0'
'$VENV/bin/pip' install -q pytorch-lightning==2.4.0 hf_transfer
'$VENV/bin/python' -c "import nemo, torch; print('nemo', nemo.__version__, 'torch', torch.__version__)"
ENV
      ;;
    fetch) cat <<FETCH
cd '$REPO/tools' || exit 1
'$VENV/bin/python' -c "from pathlib import Path; from tadabur.fastconformer_screen.decode import fetch_nemo; print(fetch_nemo(Path('$WORK')))"
FETCH
      ;;
    stage-ship-a|stage-edits) stage_set=${step#stage-}; cat <<STAGE
source '$CONDA' && conda activate '$ENV_NAME'
cd '$REPO/tools' || exit 1
python -u - <<'PY'
import json
from pathlib import Path

from tadabur.dataset_source import canonical_surah_ayah
from tadabur.shard_reader import iter_shard_rows
from tadabur.staged_audio import IndexRow, load_staged_clips, stage_clips, verify_staged
from tadabur.synthetic_edits import regenerate_clips

stage_set = "$stage_set"
ROOT = Path("$ROOT")
AUDIO = Path("$CLIPS")
AUDIO.mkdir(parents=True, exist_ok=True)
items = [json.loads(line) for line in
         (Path("tadabur/synthetic_edits/edits.jsonl")).read_text(encoding="utf-8").splitlines()
         if line.strip()]
def ship_a_clips() -> set[str]:
    from tadabur.fastconformer_screen.sets import decode_plan

    return {key.split("@", 1)[0] for key in decode_plan("ship-a")}


need = ship_a_clips() if "$stage_set" == "ship-a" else regenerate_clips(items)
registry = load_staged_clips()
missing = {name for name in need if not (AUDIO / name).exists()}
print(f"{stage_set}: {len(need)} clips, {len(missing)} to stage", flush=True)
index = {name: IndexRow(name, clip.shard, clip.row_index, clip.reciter_id,
                        clip.surah_ayah, 0.0)
         for name, clip in registry.items() if name in missing}
uses = {"ship-a": "ship_a_screen", "edits": "synthetic_edit"}
requests = {name: frozenset({uses[stage_set]}) for name in missing}

def shard_rows(shard):
    return iter_shard_rows([shard], cache_dir=ROOT / "shard_cache", delete_after=True,
                           columns=["audio", "reciter_id"])

def checkpoint(staged):
    print(f"  {len(staged)}/{len(missing)} staged", flush=True)

staged, unlocatable = stage_clips(requests, index, AUDIO, shard_rows,
                                  {n: registry[n] for n in need if n in registry}, checkpoint)
print(f"staged {len(staged)}, unlocatable {len(unlocatable)}", flush=True)
if unlocatable:
    raise SystemExit(f"unlocatable: {unlocatable[:5]}")
for name in sorted(need):
    verify_staged(registry[name], AUDIO)
print("every edit clip verifies against the registry", flush=True)
PY
if [ "$stage_set" = edits ]; then
  python -u -m tadabur.synthetic_edits generate --audio-dir '$CLIPS' --out-dir '$EDITS'
  python -u -m tadabur.synthetic_edits verify --manifest tadabur/synthetic_edits/edits.jsonl --audio-dir '$EDITS/audio'
fi
STAGE
      ;;
    decode) cat <<DECODE
cd '$REPO/tools' || exit 1
set -o pipefail
for set_name in ship-a edits qdat; do
  case "\$set_name" in
    ship-a) dir='$CLIPS' ;;
    edits) dir='$EDITS/audio' ;;
    qdat) dir='$ROOT/qdat_cache' ;;
  esac
  HF_HUB_ENABLE_HF_TRANSFER=1 '$VENV/bin/python' -u -m tadabur.fastconformer_screen.decode \
      --set "\$set_name" --dir "\$dir" --out-dir '$ROOT/decodes' \
      --nemo '$WORK/stt_ar_fastconformer_hybrid_large_pcd_v1.0.nemo' \
      --device cpu --pause-file '$PAUSE' || exit \$?
  gzip -9 -f '$ROOT/decodes/'"\$set_name"'.fastconformer.jsonl'
  gzip -9 -f '$ROOT/decodes/'"\$set_name"'.fastconformer.timing.jsonl'
done
'$VENV/bin/python' -c "from pathlib import Path; from tadabur.fastconformer_screen.decode import NOTICE; Path('$ROOT/decodes/NOTICE').write_text(NOTICE, encoding='utf-8')"
DECODE
      ;;
  esac
}

write_runner() {
  remote "cat > '$ROOT/run-$LABEL.sh' <<'SH'
#!/usr/bin/env bash
set -uo pipefail
$(command_for)
code=\$?
echo \"\$code\" > '$LOGS/$LABEL.exit'
exit \$code
SH
chmod +x '$ROOT/run-$LABEL.sh'"
}

launch() {
  require_disk
  stage_repo
  write_runner
  remote "rm -f '$LOGS/$LABEL.exit' '$LOGS/$LABEL.pid'
          setsid bash '$ROOT/run-$LABEL.sh' </dev/null >'$LOGS/$LABEL.log' 2>&1 &
          echo \$! > '$LOGS/$LABEL.pid'"
  echo "step $LABEL started on $HOST"
}

check() {
  if remote "test -f '$LOGS/${1:-$LABEL}.exit'"; then
    echo "step ${1:-$LABEL} exited $(remote "cat '$LOGS/${1:-$LABEL}.exit'")"
  else
    echo "step ${1:-$LABEL} is running (pid $(remote "cat '$LOGS/${1:-$LABEL}.pid' 2>/dev/null || echo unknown"))"
  fi
  remote "tail -n 8 '$LOGS/${1:-$LABEL}.log' 2>/dev/null || true"
}

fetchout() {
  local destination=${1:?fetchout needs a destination directory}
  mkdir -p "$destination"
  rsync -a "$HOST:$REPO/$OUT/decodes/" "$destination/"
  echo "fetched the decodes, identities and timings into $destination"
}

case "$action" in
  check) check "${3:-}" ;;
  print) command_for ;;
  fetchout) fetchout "${3:-}" ;;
  *) launch ;;
esac
