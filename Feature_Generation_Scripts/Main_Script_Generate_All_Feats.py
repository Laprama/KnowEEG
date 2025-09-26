# Run this script to generate all features 
import subprocess

scripts = ["Generate_Epoched_Connectivity_Feats.py",
    "Generate_One_Epoch_Connectivity_Feats.py",
    "Generate_Pearson_Spearman_Connectivity_Feats.py",
    "Generate_TS_Fresh_Feats.py"]

for script in scripts:
    print(f"\n>>> Running {script}...")
    subprocess.run(["python", script], check=True)
    print(f"<<< Finished {script}\n")