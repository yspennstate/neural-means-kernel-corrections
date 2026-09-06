#!/bin/bash
# Pull-based kill switch for the Caltech DGX (user yitz). Installed once as a cron job
# (* * * * * /raid/yitz/killswitch/dgx_killswitch.sh), it needs no inbound connection:
# every minute the box FETCHES one small file over HTTPS and obeys its first word.
#
#   STOP    kill every process of this user except the switch itself and sshd/tmux/bash
#   PAUSE   SIGSTOP the same set (freeze; a later RUN resumes them with SIGCONT)
#   RUN     nothing (or SIGCONT after a PAUSE)
#
# The owner's single action is therefore: edit one line in the control file on GitHub
# (repo yspennstate/neural-means-kernel-corrections, branch f5/gpu-experiments-20260906,
# file experiments/gpu_20260906/tools/DGX_CONTROL) from a phone - the box obeys within a minute
# even while sshd is not accepting logins, as long as user space still schedules.
#
# Dead-man rules (no network needed), OFF unless the control file's second line reads
# "DEADMAN on": if MemAvailable stays under MEM_MIN_GB for three consecutive minutes, or the
# local sshd stops answering on 127.0.0.1:22 for five minutes, the switch kills the user's jobs
# on its own and writes why to the log. Off by default so a watcher that can act never
# medicates a healthy box on a transient reading.
#
# The switch is a few kilobytes of shell + curl; the OOM killer takes the largest process
# first, so under thrash the offending job dies before this does.

URL="${DGX_CONTROL_URL:-https://raw.githubusercontent.com/yspennstate/neural-means-kernel-corrections/f5/gpu-experiments-20260906/experiments/gpu_20260906/tools/DGX_CONTROL}"
STATE_DIR="${STATE_DIR:-/raid/yitz/killswitch}"
LOG="$STATE_DIR/killswitch.log"
MEM_MIN_GB="${MEM_MIN_GB:-20}"
ME=$$
mkdir -p "$STATE_DIR"

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

# processes of this user that are jobs, not infrastructure
job_pids() {
    ps -u "$(id -un)" -o pid=,comm= | awk -v me="$ME" '
        $1 != me && $2 !~ /^(sshd|bash|sh|tmux|tmux:|screen|cron|curl|awk|ps|sleep|dgx_killswitch)/ { print $1 }'
}

act() {   # act STOP|PAUSE|CONT reason
    local sig="$1" why="$2" n=0
    for p in $(job_pids); do
        case "$sig" in
            STOP)  kill -TERM "$p" 2>/dev/null && n=$((n+1)) ;;
            PAUSE) kill -STOP "$p" 2>/dev/null && n=$((n+1)) ;;
            CONT)  kill -CONT "$p" 2>/dev/null && n=$((n+1)) ;;
        esac
    done
    if [ "$sig" = STOP ]; then sleep 10; for p in $(job_pids); do kill -KILL "$p" 2>/dev/null; done; fi
    log "$sig $n processes ($why); MemAvailable $(awk '/MemAvailable/ {printf "%.0f", $2/1048576}' /proc/meminfo) GB"
}

# 1. the remote word (one fetch, 20 s budget; a failed fetch is RUN, never STOP)
ctrl=$(curl -fsS --max-time 20 "$URL" 2>/dev/null)
word=$(printf '%s\n' "$ctrl" | awk 'NR==1 {print toupper($1)}')
deadman=$(printf '%s\n' "$ctrl" | awk 'NR==2 && toupper($1)=="DEADMAN" {print tolower($2)}')
case "$word" in
    STOP)  [ -f "$STATE_DIR/.stopped" ] || { act STOP "remote STOP"; touch "$STATE_DIR/.stopped"; } ;;
    PAUSE) [ -f "$STATE_DIR/.paused" ] || { act PAUSE "remote PAUSE"; touch "$STATE_DIR/.paused"; } ;;
    RUN|"") if [ -f "$STATE_DIR/.paused" ]; then act CONT "remote RUN after PAUSE"; rm -f "$STATE_DIR/.paused"; fi
            rm -f "$STATE_DIR/.stopped" ;;
esac

# 2. dead-man rules only when the owner turned them on in the control file
[ "$deadman" = "on" ] || exit 0
avail=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
if [ "$avail" -lt "$MEM_MIN_GB" ]; then
    c=$(( $(cat "$STATE_DIR/.lowmem" 2>/dev/null || echo 0) + 1 )); echo "$c" > "$STATE_DIR/.lowmem"
    [ "$c" -ge 3 ] && { act STOP "MemAvailable ${avail} GB < ${MEM_MIN_GB} GB for $c minutes"; echo 0 > "$STATE_DIR/.lowmem"; }
else
    echo 0 > "$STATE_DIR/.lowmem"
fi

# 3. dead-man: local sshd not answering (the failure of 6 Sep 2026)
if ! timeout 8 bash -c 'exec 3<>/dev/tcp/127.0.0.1/22 && read -t 5 -r line <&3 && [ -n "$line" ]' 2>/dev/null; then
    c=$(( $(cat "$STATE_DIR/.nossh" 2>/dev/null || echo 0) + 1 )); echo "$c" > "$STATE_DIR/.nossh"
    [ "$c" -ge 5 ] && { act STOP "sshd sent no banner for $c minutes"; echo 0 > "$STATE_DIR/.nossh"; }
else
    echo 0 > "$STATE_DIR/.nossh"
fi
