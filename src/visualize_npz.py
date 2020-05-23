import numpy as np
import subprocess
import argparse
import cv2 as cv


parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, type=str)
path = parser.parse_args().input

npz = np.load(path)

data = npz["data"]
target = npz["target"]
age = npz["age"]
survival = npz["survival"]
tumor_type = npz["tumor_type"]
res = npz["res"]

print(data.shape, target.shape, age, survival, tumor_type, res)

display_str = f"Age: {age}, survival: {survival}, tumor_type: {tumor_type}, res: {res}"

target = target.astype(np.float32)
for s in range(data.shape[-1]):
    display_data = np.hstack((data[0, :, :, s], data[1, :, :, s], data[2, :, :, s], data[3, :, :, s]))
    target_display = np.hstack((target[0, :, :, s], target[1, :, :, s], target[2, :, :, s], target[3, :, :, s]))
    display = np.vstack((display_data, target_display))

    cv.imshow(display_str, display)

    key = cv.waitKey(0)
    if key == 27:
        break
