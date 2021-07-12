FROM ubuntu:18.04

##############################################################################
#
#   General tools
#
##############################################################################

RUN apt-get update && \
    apt-get install -y \
        wget \
        unzip \
        build-essential \
        cmake \
        git \
        pkg-config \
        autoconf \
        automake \
        git-core \
        python3-dev \
        python3-pip \
        python3-numpy \
        python3-pkgconfig && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s $(which python3) /usr/local/bin/python

##############################################################################
#
#   Pangolin Viewer
#
##############################################################################

# Install pangoling dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        pkg-config \
        libgl1-mesa-dev \
        libglew-dev \
        cmake \
        libpython2.7-dev \
        libegl1-mesa-dev \
        libwayland-dev \
        libxkbcommon-dev \
        wayland-protocols \
        libeigen3-dev \
        doxygen && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install pyopengl Pillow pybind11

WORKDIR /home
RUN git clone https://github.com/LukasBommes/Pangolin.git pangolin

WORKDIR /home/pangolin
RUN git submodule init && git submodule update

WORKDIR /home/pangolin/build
RUN cmake .. && \
    cmake --build . && \
    cmake --build . --target doc

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics


##############################################################################
#
#   pyg2o hyper graph optimizer
#
##############################################################################

RUN apt-get update && \
  apt-get install -y \
    software-properties-common \
    lsb-release \
    mesa-utils \
    wget \
    curl \
    sudo vim \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
  apt-get install -y \
    cmake \
    git \
    build-essential \
    libeigen3-dev \
    libsuitesparse-dev \
    qtdeclarative5-dev \
    qt5-qmake \
    libqglviewer-dev-qt4 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
  apt-get install -y \
    python-dev \
    python3-dev \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /code
RUN git clone https://github.com/LukasBommes/g2opy.git

WORKDIR /code/g2opy/build
  RUN cmake .. \
  && make -j12 \
  && make install -j12 \
  && ldconfig

WORKDIR /code/g2opy/
RUN python setup.py install

WORKDIR /code/g2opy


##############################################################################
#
#   Python Packages
#
##############################################################################

# needed for opencv-python and matplotlib
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        libsm6 \
        libxext6 \
        libxrender-dev \
        python3-tk \
    	libcanberra-gtk-module \
    	libcanberra-gtk3-module && \
	rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt /
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r /requirements.txt


##############################################################################
#
#   Container Startup
#
##############################################################################

WORKDIR /dronemapper

EXPOSE 8888

CMD ["sh", "-c", "tail -f /dev/null"]
