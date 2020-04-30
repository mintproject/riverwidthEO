FROM osgeo/gdal:ubuntu-small-latest

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get upgrade -y --allow-unauthenticated && \
    apt-get install -y --allow-unauthenticated \
    python3-pip \
    libspatialindex-dev \
    wget \
    ffmpeg \
    nano \
    vim \
    zip

WORKDIR /block
COPY requirements.txt /block
RUN pip3 install -r requirements.txt
RUN pip3 install --upgrade cloudpickle
