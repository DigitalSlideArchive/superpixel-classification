# FROM python:3.9-slim
FROM tensorflow/tensorflow:latest-gpu

# This tells girder_worker to enable gpu if possible
LABEL com.nvidia.volumes.needed=nvidia_driver

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    rdfind \
    && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /var/cache/*

COPY . /opt/scw
WORKDIR /opt/scw
RUN python -m pip install --no-cache-dir -e .[tensorflow,torch] --find-links https://girder.github.io/large_image_wheels --extra-index-url https://download.pytorch.org/whl/cu126 && \
    rm -rf /root/.cache/pip/* && \
    rdfind -minsize 32768 -makehardlinks true -makeresultsfile false /usr/local

# Use a newer histomicstk
# Not needed if we install histomicstk from pypi
# RUN apt-get update && apt-get install -y git build-essential && \
#     git clone --depth=1 --single-branch -b master https://github.com/DigitalSlideArchive/HistomicsTK.git && \
#     cd HistomicsTK && \
#     pip install .[tensorflow,torch]

WORKDIR /opt/scw/superpixel_classification

RUN python -m slicer_cli_web.cli_list_entrypoint --list_cli
RUN python -m slicer_cli_web.cli_list_entrypoint SuperpixelClassification --help

# This makes the results show up in a more timely manner
ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]
