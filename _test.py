import subprocess
import os

def test_dvc_repro_runs_successfully():
    project_path = os.path.join(os.getcwd(), "ONLINE-FOOD-FEEDBACK")

    result = subprocess.run(
        ["dvc", "repro"],
        cwd=project_path,
        capture_output=True,
        text=True
    )

    # Always print logs so GitHub Actions shows them
    print("=== STDOUT ===")
    print(result.stdout)
    print("=== STDERR ===")
    print(result.stderr)

    # Assert DVC ran successfully
    assert result.returncode == 0, "dvc repro failed"
