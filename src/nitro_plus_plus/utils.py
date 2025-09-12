import argparse, sys, subprocess


def kill_port(port: int) -> int:
    """Kill any process(es) listening on the given TCP `port` (Windows)."""
    # Windows PowerShell approach
    try:
        cmd = [
            "powershell", "-NoProfile", "-Command",
            f"(Get-NetTCPConnection -State Listen -LocalPort {port} -ErrorAction SilentlyContinue | "
            f"Select-Object -Expand OwningProcess -Unique) -join ' '"
        ]
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        if not out:
            print(f"No listener on port {port}.")
            return 0
        for pid in [p for p in out.split() if p.isdigit()]:
            subprocess.run(["taskkill", "/PID", pid, "/F"], check=False)
            print(f"Killed PID {pid} on port {port}.")
        return 0
    except Exception as e:
        print(f"Failed to kill port {port}: {e}", file=sys.stderr)
        return 1


def kill_port_main():
    """Console script for `killport <port>`."""
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    args = ap.parse_args()
    sys.exit(kill_port(args.port))
