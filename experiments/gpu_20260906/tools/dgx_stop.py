"""One command that stops every job of user yitz on the Caltech DGX, through whichever channel
opens first.

    python dgx_stop.py            STOP: flip the pulled control word to STOP (works while sshd is hung,
                                  once the switch is installed on the box), then keep trying SSH - one
                                  attempt every 30 s - and on the first login kill all of yitz's jobs,
                                  install the pull switch (cron, every minute) and report what died.
    python dgx_stop.py --pause    same with PAUSE (SIGSTOP; --run resumes)
    python dgx_stop.py --run      flip the word back to RUN (and, if SSH answers, resume paused jobs)
    python dgx_stop.py --install  only install/refresh the switch on the box (no kill)

The password comes from C:/Users/owner/.claude/dreaming/secrets/caltech_dgx_login.txt (owner order
of 6 Sep 2026) or from DGX_PW. The control word lives in the repository branch
f5/gpu-experiments-20260906 (experiments/gpu_20260906/tools/DGX_CONTROL) and is pushed from the
worktree C:/Users/owner/nmkc-f5-20260906; the box fetches it raw from GitHub.
"""
import os, subprocess, sys, time
from pathlib import Path
import paramiko

HOST, USER = "131.215.141.72", "yitz"
SECRETS = Path("C:/Users/owner/.claude/dreaming/secrets/caltech_dgx_login.txt")
WT = Path("C:/Users/owner/nmkc-f5-20260906")
CTRL = WT / "experiments/gpu_20260906/tools/DGX_CONTROL"
SWITCH_LOCAL = Path(__file__).resolve().parent / "dgx_killswitch.sh"
SWITCH_REMOTE = "/raid/yitz/killswitch/dgx_killswitch.sh"
LOG = Path("C:/Users/owner/GOAL20H_20260906/nmkc_gpu_experiments/logs/dgx_stop.log")


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def password():
    pw = os.environ.get("DGX_PW")
    if pw:
        return pw
    for line in SECRETS.read_text(encoding="utf-8").splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1].strip()
    sys.exit("no DGX password: set DGX_PW or write the secrets file")


def flip(word):
    """Write the control word and push it; the box pulls it within a minute."""
    lines = CTRL.read_text(encoding="utf-8").splitlines() if CTRL.exists() else ["RUN", "DEADMAN off"]
    lines[0] = word
    CTRL.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        subprocess.run(["git", "-C", str(WT), "add", str(CTRL)], check=True, capture_output=True)
        r = subprocess.run(["git", "-C", str(WT), "-c", "user.name=Yitzchak Shmalo", "-c", "user.email=yitz@proofswarmcom.com",
                            "commit", "-q", "-m", f"DGX control: {word}"], capture_output=True, text=True)
        if r.returncode == 0 or "nothing to commit" in r.stdout + r.stderr:
            p = subprocess.run(["git", "-C", str(WT), "push", "-q", "origin", "f5/gpu-experiments-20260906"],
                               capture_output=True, text=True, timeout=600)   # this laptop under load pushes slowly
            log(f"control word {word} pushed" if p.returncode == 0 else f"push failed: {p.stderr.strip()[:200]}")
    except Exception as ex:  # noqa: BLE001
        log(f"flip {word}: git error {ex!r}")


def connect(pw, timeout=600):
    """One patient socket: sshd's parent on a thrashing box accepts rarely, and every short attempt
    that gives up leaves a stale connection in its backlog for it to waste an accept on. So one
    connection waits up to ten minutes for the banner instead of thirty tries of 25 s."""
    import socket, struct
    # A connection we abandon with a clean close (FIN) sits in sshd's accept queue until the hung
    # parent accepts it. Closing with a reset instead (SO_LINGER 1,0 -> RST) is meant to let the
    # kernel discard the entry so this prober never clogs the queue it is waiting on - that is the
    # intent, not a measured fact (unverified on 6 Sep 2026); at worst it is harmless. Windows caps
    # the connect wait at its own SYN-retry limit (about two minutes) whatever timeout is asked.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    sock.settimeout(timeout)
    try:
        sock.connect((HOST, 22))
    except Exception:
        sock.close()          # RST, not FIN
        raise
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        c.connect(HOST, username=USER, password=pw, sock=sock, look_for_keys=False, allow_agent=False,
                  timeout=timeout, banner_timeout=timeout, auth_timeout=timeout)
    except Exception:
        try:
            sock.close()      # RST
        except Exception:
            pass
        raise
    return c


def sh(c, cmd, timeout=120):
    _, out, err = c.exec_command(cmd, timeout=timeout)
    o = out.read().decode("utf-8", "replace"); e = err.read().decode("utf-8", "replace")
    return out.channel.recv_exit_status(), o, e


KILL_CMD = r"""
me=$(id -un); n=0
for p in $(ps -u "$me" -o pid=,comm= | awk '$2 !~ /^(sshd|bash|sh|tmux|tmux:|screen|cron|ps|awk)/ {print $1}'); do
  [ "$p" = "$$" ] && continue; kill -%SIG% "$p" 2>/dev/null && n=$((n+1)); done
echo "signalled $n"; sleep 8
%KILL9%
echo "--- remaining:"; ps -u "$me" -o pid=,etime=,rss=,comm= | sort -k3 -n -r | head -15
echo "--- memory:"; free -g | sed -n 1,2p; uptime
"""


def kill_all(c, pause=False):
    cmd = KILL_CMD.replace("%SIG%", "STOP" if pause else "TERM")
    cmd = cmd.replace("%KILL9%", "" if pause else
                      'for p in $(ps -u "$me" -o pid=,comm= | awk \'$2 !~ /^(sshd|bash|sh|tmux|tmux:|screen|cron|ps|awk)/ {print $1}\'); do [ "$p" = "$$" ] || kill -KILL "$p" 2>/dev/null; done')
    rc, o, e = sh(c, cmd, timeout=180)
    log(("PAUSED" if pause else "KILLED") + " on the DGX:\n" + o + (("[stderr] " + e) if e.strip() else ""))


def install(c):
    s = c.open_sftp()
    sh(c, "mkdir -p /raid/yitz/killswitch")
    s.put(str(SWITCH_LOCAL), SWITCH_REMOTE)
    s.close()
    rc, o, e = sh(c, f"chmod +x {SWITCH_REMOTE}; "
                     f"(crontab -l 2>/dev/null | grep -v dgx_killswitch; echo '* * * * * {SWITCH_REMOTE} >/dev/null 2>&1') | crontab -; "
                     f"crontab -l | grep -c dgx_killswitch; bash -n {SWITCH_REMOTE} && echo switch-ok")
    log(f"switch installed (cron lines: {o.strip().replace(chr(10), ' ')}) {e.strip()[:200]}")


def main():
    a = sys.argv[1:]
    word = "PAUSE" if "--pause" in a else "RUN" if "--run" in a else None if "--install" in a else "STOP"
    pw = password()
    if word:
        flip(word)
    log(f"waiting for an SSH login to {HOST} (one patient connection at a time, banner wait 600 s; Ctrl-C to stop waiting)")
    while True:
        try:
            c = connect(pw)
        except Exception as ex:  # noqa: BLE001
            log(f"no login yet after a long wait: {str(ex)[:80]}")
            time.sleep(15)
            continue
        try:
            rc, o, _ = sh(c, "hostname; uptime")
            log("LOGIN OK: " + o.strip().replace("\n", " | "))
            if word == "STOP":
                kill_all(c)
            elif word == "PAUSE":
                kill_all(c, pause=True)
            elif word == "RUN":
                sh(c, "pkill -CONT -u $(id -un) 2>/dev/null; true")
                log("sent SIGCONT to all of yitz's processes")
            install(c)
        finally:
            c.close()
        break


if __name__ == "__main__":
    main()
