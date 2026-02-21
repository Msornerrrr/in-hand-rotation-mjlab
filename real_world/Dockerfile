FROM ros:noetic-ros-base-focal

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir numpy pyzmq

WORKDIR /workspace/in-hand-rotation-mjlab
