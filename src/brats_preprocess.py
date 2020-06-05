'''
This module reads the original BratS 2020 data and saves preprocessed .npz files, to be used by the BRATS Dataset class.

Performs the same pre-processing done by Isensee Et Al.
Mean STD with clips for outliers in [-5, 5] and subsequent normalization to [0, 1]
Nothing is done with targets, only permutation of channel posisitions.

Save specifications follow Medical Segmentation Decatlhon JSON

"modality": {
     "0": "FLAIR",
     "1": "T1w",
     "2": "t1gd",
     "3": "T2w"
 },
 "labels": {
     "0": "background",
     "1": "edema",
     "2": "non-enhancing tumor",
     "3": "enhancing tumour"


     "originals": {
     "0": "background",
     "1": "non-enhancing tumor",
     "2": "edema",
     "4": "enhancing tumour"
 }
'''
import glob
import os
import numpy as np
import nibabel as nib
import multiprocessing as mp
import nibabel
import pandas as pd
import mlflow
import argparse
import datetime
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default="/home/diedre/Dropbox/bigdata/brats/2020/train")
args = parser.parse_args()
DATA_PATH = args.data_path

assert os.path.isdir(DATA_PATH), "DATA_PATH {DATA_PATH} is not a folder."


def worker(subject):
    folder = os.path.join(DATA_PATH, subject)

    survival_csv = pd.read_csv(os.path.join(DATA_PATH, "survival_info.csv"))
    grade_csv = pd.read_csv(os.path.join(DATA_PATH, "name_mapping.csv"))

    try:
        survival_row = survival_csv.loc[survival_csv['Brats20ID'] == subject]

        survival = survival_row['Survival_days'].values[0]
        age = survival_row['Age'].values[0]
        res = survival_row['Extent_of_Resection'].values[0]
    except Exception:
        survival, age, res = 'unk', 'unk', 'unk'

    try:
        tumor_type = grade_csv.loc[grade_csv['BraTS_2020_subject_ID'] == subject]['Grade'].values[0]
    except Exception:
        tumor_type = 'unk'

    log = f"Survival: {survival}, age: {age}, res: {res}, tumor_type: {tumor_type}"

    dst = folder
    assert os.path.isdir(dst)

    path = {}
    for key in keys:
        search_for_file_in_folder(path, dst, key)

    log += f"\n\nDetected paths: {path}\n"

    save_data = None
    save_seg = 'unk'
    for key, path in path.items():
        data = nib.load(path).get_fdata()
        if save_data is None:
            save_data = np.zeros((4,) + data.shape, dtype=data.dtype)

        if key == "seg":
            # Segmentation is ints, converting to one hot (original max value = 4)
            seg = np.eye(5)[data.astype(np.int)].astype(np.int).transpose(3, 0, 1, 2)  # put channel dim in beginning
            save_seg = np.zeros((4,) + data.shape, dtype=np.int)
            save_seg[0] = seg[0]
            save_seg[1] = seg[2]
            save_seg[2] = seg[1]
            save_seg[3] = seg[4]
        else:
            # Isensee brain normalization
            # Compute statistics only in brain region, ignoring zeros
            nan_data = data.copy()
            nan_data[nan_data == 0] = np.nan
            mean_of_brain = np.nanmean(nan_data)
            std_of_brain = np.nanstd(nan_data)

            data = (data - mean_of_brain) / std_of_brain
            data[data > 5.0] = 5.0
            data[data < -5.0] = -5.0
            data = (data - data.min()) / (data.max() - data.min())
            save_data[keys.index(key)] = data

    save_name = os.path.join(dst, f"{os.path.basename(dst)}_preprocessed.npz")

    np.savez_compressed(save_name, data=save_data, target=save_seg, tumor_type=tumor_type, age=age, survival=survival, res=res)

    log += f"\nSaved in: {save_name}\n\n"

    return log


def search_for_file_in_folder(dict_ref, folder_path, key):
    try:
        path = glob.glob(os.path.join(folder_path, f"*{key}*"))[0]
        dict_ref[key] = path
    except Exception:
        if key == "seg":
            return
        else:
            raise ValueError(f"Didn't find file corresponding to key {key}")


if __name__ == "__main__":
    keys = ["flair", "t1", "t1ce", "t2", "seg"]

    paths = []
    folder_list = []

    subjects = ["BraTS20_Training_" + str(i).zfill(3) for i in range(1, 370)]

    cpu_count = max(mp.cpu_count() - 1, 1)
    pool = mp.Pool(processes=cpu_count)
    logs = 'Logs for pre_process run\n\n'

    print(f"Pre processing with {cpu_count} cores...")
    for log in tqdm(pool.imap_unordered(worker, subjects), total=len(subjects), leave=True, position=0):
        logs += log
    print("Done.")

    os.makedirs("logs", exist_ok=True)  # for safety
    logpath = "logs/" + str(datetime.datetime.now()) + ".txt"

    with open(logpath, 'w') as logfile:
        logfile.write(logs)

    mlflow.log_artifact(logpath)
