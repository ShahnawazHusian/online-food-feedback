import subprocess
import os

def test_dvc_repro_runs_successfully():
    # Go to the folder where dvc.yaml is located
    project_path = os.path.join(os.getcwd(), "online food")  # Adjust if needed

    # Run `dvc repro`
    result = subprocess.run(
        ["dvc", "repro"],
        cwd=project_path,
        capture_output=True,
        text=True
    )

    # Debug info (only prints if test fails)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    # Assert DVC ran successfully
    assert result.returncode == 0, "dvc repro failed"