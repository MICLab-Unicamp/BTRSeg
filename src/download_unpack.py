import os
import sys
import subprocess


if not os.path.isfile(os.path.join("data", "data.zip")):
    print("Original data not found, downloading it (3GB, will take some minutes)...")
    subprocess.run(["wget", "https://ndownloader.figshare.com/files/22813406?private_link=466e909eea5a3e8ec1e3",
                    "-O", "data/data.zip"], stdout=sys.stdout, stderr=sys.stderr)
else:
    print("Original data already downloaded.")

if os.path.isdir("data/MICCAI_BraTS2020_TrainingData"):
    print("Original data already unpacked.")
else:
    print("Decompressing original data...")
    subprocess.run(["unzip", "-q", "data/data.zip", "-d", "data"], stdout=sys.stdout, stderr=sys.stderr)
    print("Done.")
