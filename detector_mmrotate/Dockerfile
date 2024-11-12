FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

RUN apt-get update
RUN apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6

RUN pip install openmim numpy\<2.0
RUN mim install mmcv-full==1.7.2
RUN mim install mmdet\<3.0.0

COPY /. /opt/mmrotate/
RUN pip install -v -e /opt/mmrotate/
WORKDIR /opt/mmrotate/
