import mlflow
import numpy as np
import argparse
import cv2 as cv

print("Generating data visualization...")

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--no_display', action='store_true')
args = parser.parse_args()
path = args.input
no_display = args.no_display

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

if no_display:
    s = 60
    display_data = np.hstack((data[0, :, :, s], data[1, :, :, s], data[2, :, :, s], data[3, :, :, s]))
    target_display = np.hstack((target[0, :, :, s], target[1, :, :, s], target[2, :, :, s], target[3, :, :, s]))
    display = np.vstack((display_data, target_display))

    display = (display * 255).astype(np.uint8)

    cv.imwrite("figures/display.png", display)
    mlflow.log_artifact("figures/display.png")
    print("Visualization saved in figures/display.png")
else:
    for s in range(data.shape[-1]):
        display_data = np.hstack((data[0, :, :, s], data[1, :, :, s], data[2, :, :, s], data[3, :, :, s]))
        target_display = np.hstack((target[0, :, :, s], target[1, :, :, s], target[2, :, :, s], target[3, :, :, s]))
        display = np.vstack((display_data, target_display))

        cv.imshow(display_str, display)

        key = cv.waitKey(0)
        if key == 27:
            break
