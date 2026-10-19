"""The `hydroknight` dispatcher: help lists every command, and every command's own
`--help` works without the optional extras (CI installs neither MLflow nor MediaPipe)."""

import subprocess
import sys
import textwrap
from pathlib import Path

from hydro_knight import cli

CLI_DIR = Path(cli.__file__).parent


def test_help_lists_every_command(capsys):
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    for c in cli.COMMANDS:
        assert c.name in out and c.help in out


def test_every_cli_module_is_a_command():
    modules = {p.stem for p in CLI_DIR.glob("*.py")} - {"__init__", "__main__"}
    assert modules == {c.module for c in cli.COMMANDS}


def test_group_help_lists_only_its_choices(capsys):
    assert cli.main(["train", "--help"]) == 0
    out = capsys.readouterr().out
    assert "train tcn-v2" in out and "eval protocol" not in out


def test_unknown_command_suggests_the_close_one(capsys):
    assert cli.main(["trian"]) == 2
    assert "Did you mean 'train'?" in capsys.readouterr().err
    assert cli.main(["eval", "nope"]) == 2


def test_runs_module_with_remaining_args(monkeypatch):
    seen = {}

    class Fake:
        @staticmethod
        def main():
            seen["argv"] = list(sys.argv)

    monkeypatch.setattr(cli.importlib, "import_module", lambda name: Fake)
    monkeypatch.setattr(sys, "argv", ["hydroknight"])
    assert cli.main(["eval", "protocol", "runs/x", "--pseudo-normal"]) == 0
    assert seen["argv"] == ["hydroknight eval protocol", "runs/x", "--pseudo-normal"]


def test_help_everywhere_with_extras_blocked():
    # A fresh interpreter, so blocking the extras can't be undone by an earlier import.
    code = textwrap.dedent(
        """
        import sys
        for extra in ("mlflow", "mediapipe"):
            sys.modules[extra] = None  # `import mlflow` now raises ImportError
        from hydro_knight import cli
        assert cli.main(["--help"]) == 0
        for c in cli.COMMANDS:
            try:
                cli.main([*c.name.split(), "--help"])
            except SystemExit as e:
                assert e.code == 0, (c.name, e.code)
            else:
                raise AssertionError(f"{c.name} --help did not exit")
        print("ok")
        """
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-2000:]
