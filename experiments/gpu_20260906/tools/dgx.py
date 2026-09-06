"""Caltech DGX helper (owhadi-lab-dgx-station, 131.215.141.72, user yitz, password auth).

The password comes ONLY from the environment variable DGX_PW (never from a file in the repo,
never printed, never on the mesh). One authentication attempt per call (lockout etiquette from
memory reference_caltech_dgx_access_2026_08_28), look_for_keys=False, allow_agent=False.

    python dgx.py run "<command>"            run a shell command, print stdout/stderr, exit with its status
    python dgx.py put <local> <remote>       sftp upload one file (creates the remote parent dir)
    python dgx.py get <remote> <local>       sftp download one file
"""
import os, sys, time
import paramiko

HOST, USER = "131.215.141.72", "yitz"


def connect(timeout=60):
    pw = os.environ.get("DGX_PW")
    if not pw:
        sys.exit("DGX_PW not set")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=pw, look_for_keys=False, allow_agent=False,
              timeout=timeout, banner_timeout=timeout, auth_timeout=timeout)
    return c


def run(cmd, timeout=600):
    c = connect()
    try:
        _, out, err = c.exec_command(cmd, timeout=timeout, get_pty=False)
        o = out.read().decode("utf-8", "replace"); e = err.read().decode("utf-8", "replace")
        rc = out.channel.recv_exit_status()
        sys.stdout.write(o)
        if e:
            sys.stdout.write("[stderr] " + e)
        return rc
    finally:
        c.close()


def put(local, remote):
    c = connect()
    try:
        s = c.open_sftp()
        d = os.path.dirname(remote)
        if d:
            c.exec_command(f"mkdir -p '{d}'")[1].channel.recv_exit_status()
        t0 = time.time(); s.put(local, remote)
        print(f"put {local} -> {remote} ({os.path.getsize(local)/1e6:.1f} MB, {time.time()-t0:.0f} s)")
        s.close()
    finally:
        c.close()


def get(remote, local):
    c = connect()
    try:
        s = c.open_sftp(); s.get(remote, local); s.close(); print(f"got {remote} -> {local}")
    finally:
        c.close()


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    if a[0] == "run":
        sys.exit(run(" ".join(a[1:])))
    if a[0] == "put":
        put(a[1], a[2])
    elif a[0] == "get":
        get(a[1], a[2])
    else:
        sys.exit(__doc__)
