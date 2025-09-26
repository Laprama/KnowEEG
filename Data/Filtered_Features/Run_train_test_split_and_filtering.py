# Run this script to do train-test split, connectivity parameter selection and TSFresh Feature Filtering
import subprocess


scripts = ["Train_Test_Split.py",
    "Filter_TS_Fresh_features.py",
    "Connectivity_Local_Parameter_Selection.py"]


for script in scripts:
    print(f"\n>>> Running {script}...")
    subprocess.run(["python", script], check=True)
    print(f"<<< Finished {script}\n")

