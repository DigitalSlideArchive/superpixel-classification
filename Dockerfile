# FROM python:3.9-slim
FROM tensorflow/tensorflow:latest-gpu

RUN python -m pip install histomicstk --find-links https://girder.github.io/large_image_wheels

# Use a newer histomicstk
# Not needed if we install histomicstk from pypi
# RUN apt-get update && apt-get install -y git build-essential && \
#     git clone --depth=1 --single-branch -b master https://github.com/DigitalSlideArchive/HistomicsTK.git && \
#     cd HistomicsTK && \
#     pip install .
RUN python -m pip install --pre 'histomicstk>=1.2.2.dev7'

RUN python -m pip install girder-client girder-slicer-cli-web h5py tensorflow keras

COPY . /opt/scw
WORKDIR /opt/scw/cli

RUN python -m slicer_cli_web.cli_list_entrypoint --list_cli
RUN python -m slicer_cli_web.cli_list_entrypoint SuperpixelClassification --help

ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]
