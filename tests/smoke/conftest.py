import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parents[2]
STUBBED_APP = Path(__file__).resolve().parent / "stubbed_app.py"
STARTUP_TIMEOUT_S = 60

# Variables the apps read. Cleared for every server so a developer's real
# credentials never leak into the smoke run.
APP_ENV_VARS = [
    "OPENAI_API_KEY",
    "DB_CONN_STR",
    "DB_USERNAME",
    "DB_PASSWORD",
    "DB_BUCKET",
    "DB_SCOPE",
    "DB_COLLECTION",
    "CACHE_COLLECTION",
    "INDEX_NAME",
    "AUTH_ENABLED",
    "LOGIN_PASSWORD",
]

# Placeholder values for the stubbed run; nothing connects to them.
PLACEHOLDER_ENV = {name: f"smoke-{name.lower()}" for name in APP_ENV_VARS}
PLACEHOLDER_ENV["DB_CONN_STR"] = "couchbase://smoke.invalid"
PLACEHOLDER_ENV["AUTH_ENABLED"] = "False"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class StreamlitServer:
    def __init__(self, script_args, extra_env, tmp_home: Path):
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.log_path = tmp_home / "streamlit.log"
        env = {k: v for k, v in os.environ.items() if k not in APP_ENV_VARS}
        env.update(extra_env)
        # Isolated HOME so ~/.streamlit/secrets.toml and config are not picked up.
        env["HOME"] = str(tmp_home)
        script, *script_argv = script_args
        cmd = [
            sys.executable, "-m", "streamlit", "run", script,
            "--server.headless", "true",
            "--server.port", str(self.port),
            "--server.address", "127.0.0.1",
            "--server.fileWatcherType", "none",
            "--browser.gatherUsageStats", "false",
        ]
        if script_argv:
            cmd += ["--", *script_argv]
        self._log = open(self.log_path, "w")
        self.proc = subprocess.Popen(
            cmd, cwd=REPO_ROOT, env=env, stdout=self._log, stderr=subprocess.STDOUT
        )
        self._wait_healthy()

    def _wait_healthy(self):
        deadline = time.time() + STARTUP_TIMEOUT_S
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"Streamlit exited early:\n{self.log()}")
            try:
                with urllib.request.urlopen(f"{self.url}/_stcore/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except OSError:
                pass
            time.sleep(0.5)
        raise RuntimeError(f"Streamlit not healthy after {STARTUP_TIMEOUT_S}s:\n{self.log()}")

    def log(self) -> str:
        self._log.flush()
        return self.log_path.read_text()

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self._log.close()


@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def page(browser):
    context = browser.new_context()
    pg = context.new_page()
    yield pg
    context.close()


@pytest.fixture
def start_app(tmp_path):
    """Start an app: start_app("chat_with_pdf_query.py", stubbed=True)."""
    servers = []

    def _start(entrypoint: str, stubbed: bool, env=None):
        if stubbed:
            args = [str(STUBBED_APP), entrypoint]
            extra_env = dict(PLACEHOLDER_ENV)
        else:
            args = [entrypoint]
            extra_env = {}
        extra_env.update(env or {})
        home = tmp_path / f"home{len(servers)}"
        home.mkdir()
        server = StreamlitServer(args, extra_env, home)
        servers.append(server)
        return server

    yield _start
    for server in servers:
        server.stop()
