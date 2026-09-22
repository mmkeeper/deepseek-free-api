import subprocess, sys, time, os, io

LOGFILE = os.path.join("logs", "exit.log")
ERRFILE = os.path.join("logs", "server.err")
os.makedirs("logs", exist_ok=True)

cmd = [sys.executable, "server.py", "--port", "18632", "--host", "127.0.0.1", "--debug", "--no-search"]

start = time.time()
entry = {"pid": os.getpid(), "wrapper_start": time.strftime("%Y-%m-%d %H:%M:%S")}

with open(ERRFILE, "wb") as errf:
    p = subprocess.Popen(cmd, stdout=errf, stderr=errf)
    entry["server_pid"] = p.pid
    rc = p.wait()

entry["exit_code"] = rc
entry["exit_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
entry["uptime_secs"] = round(time.time() - start, 1)

with open(LOGFILE, "a", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    for k, v in entry.items():
        f.write(f"{k}: {v}\n")
    with open(ERRFILE, "rb") as er:
        er.seek(0, io.SEEK_END)
        size = er.tell()
        er.seek(max(0, size - 2048))
        tail = er.read().decode("utf-8", "replace")
    f.write("--- server.err tail ---\n")
    f.write(tail[-2048:])
    f.write("\n")