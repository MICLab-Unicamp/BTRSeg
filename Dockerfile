# Base image: CUDA and CUDNN configured for Ubuntu 18.04
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
COPY requirements.txt /
WORKDIR /

# Update Ubuntu 18 and install Python in the correct version.
RUN apt-get update && apt-get upgrade -y
RUN apt-get install python3.6 python3.6-dev -y
RUN ln -sf /usr/bin/python3.6 /usr/bin/python3

# Install pip and install python enviroment
RUN apt-get install python3-pip -y
RUN python3 -m pip install pip --upgrade
RUN python3 -m pip install -r requirements.txt

# Install AMP for mixed precision training
RUN apt-get install git -y
RUN git clone https://github.com/dscarmo/apex
RUN python3 -m pip install --no-cache-dir ./apex

WORKDIR /home/btrseg

# Build command: sudo docker build -t dscarmo/btrseg .
# Run command: sudo docker run -it --rm --gpus all dscarmo/btrseg
