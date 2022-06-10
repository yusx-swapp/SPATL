FROM anibali/pytorch:1.10.0-cuda11.3-ubuntu20.04

# Set up time zone.
ENV TZ=UTC
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

# Install system libraries required by OpenCV.
RUN sudo apt-get update \
 && sudo apt-get install -y libgl1-mesa-glx libgtk2.0-0 libsm6 libxext6 \
 && sudo rm -rf /var/lib/apt/lists/*

# Install OpenCV from PyPI.
RUN pip install opencv-python==4.5.1.48

# install conda
ENV PYTHON_VERSION=3.8
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    sh ~/miniconda.sh -b -p $HOME/conda && \
    rm ~/miniconda.sh
RUN $HOME/conda/bin/conda update -n base conda

# install requirements
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN $HOME/conda/bin/conda install conda install pyg -c pyg
