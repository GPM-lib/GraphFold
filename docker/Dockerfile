FROM nvcr.io/nvidia/cuda:11.7.0-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
     DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         apt-transport-https \
         ca-certificates \
         gnupg \
         wget && \
     rm -rf /var/lib/apt/lists/*
RUN  echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
     wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
         apt-get update -y && \
     DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         nsight-compute-2021.1.1 && \
     rm -rf /var/lib/apt/lists/*


RUN apt update \
    && apt install -y wget \
                    python3 \
                    libssl-dev \
                    libevent-core-2.1-7 \
                    libhwloc-dev \
                    libevent-pthreads-2.1-7

WORKDIR /tmp
RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4.tar.gz
RUN tar -zxvf cmake-3.26.4.tar.gz
WORKDIR /tmp/cmake-3.26.4/
RUN ./bootstrap
RUN make -j64
RUN make install


RUN ln -s /usr/bin/python3.8 /usr/bin/python
WORKDIR /GraphFold
