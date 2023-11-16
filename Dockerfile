FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

WORKDIR /root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        build-essential \
        cmake \
        ffmpeg \
        libsm6 \
        libxext6 \
        python3-pip
    
RUN pip install numpy \
                matplotlib\
                PyYAML\
                tqdm\
                torch\
                torchvision\
                opencv-python \
                scikit-image \
                kornia

RUN git clone https://github.com/larocs/modvo.git \
    && cd modvo \
    && git submodule update --init --recursive modvo/thirdparty \
    && pip install .
    

CMD ["bash"]