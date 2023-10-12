FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
ENV OPENCV_VERSION 3.2.0
ENV OPENCV_DOWNLOAD_URL https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip

ENV OpenCV_DIR opencv-$OPENCV_VERSION
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

WORKDIR /root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        build-essential \
        cmake \
        g++ wget unzip\
        ffmpeg \
        libsm6 \
        libxext6 \
        python3-pip
    
RUN pip install --no-cache-dir numpy \
                               matplotlib\
                               PyYAML\
                               tqdm\
                               torch\
                               torchvision\
                               opencv-python \
                               scikit-image \
                               evo \
                               kornia

CMD ["bash"]
