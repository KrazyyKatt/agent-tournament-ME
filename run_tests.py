import os
import subprocess

def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

print("Deleting old results.csv (if exists)...")
if os.path.exists("results.csv"):
    os.remove("results.csv")
    
with open("results.csv", "w") as f:
    f.write("blue_team,red_team,winner,reason\n")

print("Running: blu vs red (10 games)")
for i in range(10):
    print(f"  Game {i+1}")
    run("python main.py blu red --headless")
    
with open("results.csv", "a") as f:
    f.write("----,----,----,----\n")

print("Running: red vs blu (10 games)")
for i in range(10):
    print(f"  Game {i+1}")
    run("python main.py red blu --headless")

print("Done.")
print("Results saved in results.csv")
