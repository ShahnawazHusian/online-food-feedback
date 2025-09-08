import subprocess
import os

def test_dvc_repro_runs_successfully():
    # GitHub Actions will checkout the repo here
    repo_root = os.path.join(os.getcwd(), "online-food-feedback")

    # But since _test.py is already in repo root, just use cwd directly
    project_path = os.getcwd()

    result = subprocess.run(
        ["dvc", "repro"],
        cwd=project_path,
        capture_output=True,
        text=True
    )

    print("CWD:", project_path)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    assert result.returncode == 0, f"dvc repro failed\n{result.stderr}"
