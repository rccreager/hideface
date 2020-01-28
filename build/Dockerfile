FROM ubuntu:18.04
RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN apt-get install -y \
    build-essential cmake pkg-config \
    libx11-dev libatlas-base-dev \
    libgtk-3-dev libboost-python-dev \
    python3-venv s3fs  
