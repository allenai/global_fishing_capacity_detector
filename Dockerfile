FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

RUN apt-get update
RUN apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 wget


ENV PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /root/miniconda3

# conda environment for classifier
RUN conda create -y -n gfcd_classifier python=3.12
RUN git clone https://github.com/allenai/rslearn.git /opt/rslearn
RUN conda run -n gfcd_classifier pip install /opt/rslearn[extra]

# conda environment for detector
RUN conda create -y -n gfcd_detector python=3.10
RUN conda install -y -n gfcd_detector pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
RUN conda run -n gfcd_detector pip install -U openmim numpy\<2.0
RUN conda run -n gfcd_detector mim install mmcv-full==1.7.2
RUN conda run -n gfcd_detector mim install mmdet\<3.0.0

COPY ./requirements.txt /opt/gfcd/requirements.txt
RUN conda run -n gfcd_detector pip install -r /opt/gfcd/requirements.txt
RUN conda run -n gfcd_classifier pip install -r /opt/gfcd/requirements.txt

COPY ./ /opt/gfcd/
RUN conda run -n gfcd_detector pip install -v -e /opt/gfcd/detector_mmrotate/

WORKDIR /opt/gfcd/
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "gfcd_detector", "python", "-u", "docker_entrypoint.py"]
