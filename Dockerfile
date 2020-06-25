# Base image: CUDA and CUDNN configured for Ubuntu 18.04
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
COPY requirements.txt /
WORKDIR /

# Update Ubuntu 18 and install Python in the correct version.
RUN apt-get update && apt-get upgrade -y
RUN apt-get install build-essential zip unzip -y
RUN apt-get install python3.6 python3.6-dev -y
RUN ln -sf /usr/bin/python3.6 /usr/bin/python3

# Install pip and install python enviroment
RUN apt-get install python3-pip -y
RUN python3 -m pip install pip --upgrade
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN python3 -m pip install -r requirements.txt

# Install AMP for mixed precision training
RUN apt-get install git -y
RUN git clone https://github.com/dscarmo/apex
RUN python3 -m pip install --no-cache-dir ./apex

# Copy git repo/jupyter start script and setup it
COPY start.sh /home
RUN chmod +x /home/start.sh

# Copy original BraTS data
COPY data/data.zip /home
WORKDIR /home/

# Startup command
CMD ["./start.sh"]

# Build command: sudo docker build -t dscarmo/btrseg .
# Debug command: sudo docker run -it --rm --gpus all dscarmo/btrseg
# Release command: sudo docker run -p 8888:8888 --gpus all dscarmo/btrseg
# Kill running containers (stop jupyter server): sudo docker kill $(sudo docker ps -q)
# Remove dangling images: sudo docker image prune
# Remove dangling containers: sudo docker container prune