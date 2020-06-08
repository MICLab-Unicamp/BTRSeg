# BTRSeg
Repository related to the IA369Z reproducible Brain Tumor Segmentation project.

# Usage with Docker (recommended)

This work was only tested in Ubuntu 18.04. It may work in different operating systems if you follow the requirements.

## Install Docker

Follow the tutorial in: https://docs.docker.com/engine/install/ubuntu/

To use this docker image with GPU acceleration, you need a NVidia Cuda enabled GPU, with a driver version that supports CUDA 10.2.

If you don't have an appropriate GPU, you can still try to run the paper using your CPU (very slow). Check the Google Colaboratory version of the paper, where you can run some experiments using the google provided enviroment.

# Usage with Google Colab