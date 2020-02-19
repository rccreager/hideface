FROM ubuntu:18.04
RUN apt-get update
RUN apt-get install -y python3-pip \
    build-essential cmake pkg-config \
    libx11-dev libatlas-base-dev \
    libgtk-3-dev libboost-python-dev \
    python3-venv s3fs unzip
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN mkdir /build
RUN mkdir /data
RUN mkdir /hideface
COPY build/ /build
COPY data/ /data
COPY hideface/ /hideface
COPY example_*.py /
RUN pip3 install --upgrade pip && pip3 install -r /build/requirements.txt
