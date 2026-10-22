"""Start and stop a vLLM server inside the container.

Every measurement uses a fresh server process, so nothing (KV blocks, prefix cache, compiled
state that depends on earlier requests) carries over between repeats.
"""

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .common import TARGET_DIR, WORK, log


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class VllmServer:
    def __init__(self, contract, target, spec=None, name="server"):
        """spec: None for the target alone, or {"method": ..., "model": <draft dir>}."""
        self.contract = contract
        self.target = target
        self.spec = spec
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.log_path = WORK / "logs" / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}.log"
        self.proc = None

    def command(self):
        c, t = self.contract, self.target
        cmd = [
            "vllm", "serve", str(TARGET_DIR),
            "--served-model-name", "target",
            "--host", "127.0.0.1", "--port", str(self.port),
            "--dtype", t.get("dtype", "auto"),
            "--max-model-len", str(c["max_model_len"]),
            "--gpu-memory-utilization", str(c["gpu_memory_utilization"]),
            "--max-num-seqs", str(c["max_num_seqs"]),
            "--seed", str(c["seed"]),
            "--no-enable-prefix-caching",
            "--generation-config", "vllm",
            *t.get("serve_args", []),
        ]
        if self.spec:
            spec = dict(self.spec, num_speculative_tokens=c["num_speculative_tokens"])
            cmd += ["--speculative-config", json.dumps(spec)]
        return cmd

    def start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = self.command()
        log(f"starting vLLM (log: {self.log_path})")
        self._log = open(self.log_path, "w")
        self._log.write(" ".join(cmd) + "\n\n")
        self._log.flush()
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT,
                                     env=os.environ.copy(), start_new_session=True)
        t0 = time.monotonic()
        deadline = t0 + self.contract["startup_timeout_s"]
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vLLM exited with code {self.proc.returncode} during "
                                   f"startup:\n{self.tail()}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                    if r.status == 200:
                        log(f"vLLM ready after {time.monotonic() - t0:.0f}s")
                        return
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(2)
        self.stop()
        raise RuntimeError(f"vLLM not healthy after {self.contract['startup_timeout_s']}s:\n"
                           f"{self.tail()}")

    def tail(self, n=40):
        try:
            return "\n".join(Path(self.log_path).read_text().splitlines()[-n:])
        except OSError:
            return "(no log)"

    def stop(self):
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, 15)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, 9)
                self.proc.wait()
        if self.proc:
            self._log.close()
        self.proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
