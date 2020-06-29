import torch
import numpy as np
import cv2 as cv
import mlflow
from sys import argv

from train import BRATS3DSegmentation, test_transforms

print("Predicting...")
mlflow.set_experiment("Predict")

with torch.set_grad_enabled(False):
    input_path = argv[1]
    best_model = BRATS3DSegmentation.load_from_checkpoint("models/best_model.ckpt", loss=None, metric=None)
    best_model.eval()
    npz = np.load(input_path)
    data = npz["data"]

    transformed_data, _ = test_transforms(data, data)

    predicted = best_model(transformed_data.unsqueeze(0)).squeeze().detach().cpu().numpy()

    s = 60
    tdata = transformed_data.numpy()
    display_data = np.hstack((tdata[0, :, :, s], tdata[1, :, :, s], tdata[2, :, :, s], tdata[3, :, :, s]))
    target_display = np.hstack((predicted[0, :, :, s], predicted[1, :, :, s], predicted[2, :, :, s]))

    cv.imwrite("figures/predict_input.png", (display_data * 255).astype(np.uint8))
    cv.imwrite("figures/predict_output.png", (target_display * 255).astype(np.uint8))
    mlflow.log_artifact("figures/predict_input.png")
    mlflow.log_artifact("figures/predict_output.png")
    print("Prediction sample slices saved on figures/predict*.png")
