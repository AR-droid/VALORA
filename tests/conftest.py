import os

import pytest

os.environ["VALORA_OFFLINE"] = "1"
os.environ["VALORA_TICK_SECONDS"] = "0"


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("VALORA_DB", str(tmp_path / "test.db"))
    from valora.config import Settings
    from valora.runtime import Runtime, reset_runtime

    rt = Runtime(Settings())
    reset_runtime(rt)
    yield rt
    reset_runtime(None)


@pytest.fixture
def client(runtime):
    from valora.web.app import create_app

    app = create_app(runtime)
    app.config["TESTING"] = True
    return app.test_client()
