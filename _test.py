import os
import subprocess

def test_dvc_repro_runs_successfully():
    project_path = os.getcwd()   # don’t append extra folder

    result = subprocess.run(
        ["dvc", "repro"],
        cwd=project_path,
        capture_output=True,
        text=True
    )

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode == 0, f"DVC repro failed:\n{result.stderr}"
