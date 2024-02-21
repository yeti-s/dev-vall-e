FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

RUN apt-get update && \
    apt-get upgrade -y
RUN apt-get install -y vim wget git libsndfile1 espeak-ng git-lfs build-essential

RUN pip install torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install torchmetrics==0.11.1 phonemizer==3.2.1 pypinyin==0.48.0
RUN pip install lhotse librosa matplotlib


WORKDIR /workspace
RUN pip install https://huggingface.co/csukuangfj/k2/resolve/main/cuda/k2-1.24.3.dev20230718+cuda11.6.torch1.13.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


RUN mkdir dependencies
WORKDIR /workspace/dependencies
RUN git clone https://github.com/k2-fsa/icefall
WORKDIR /workspace/dependencies/icefall
RUN pip install -r requirements.txt
RUN pip install -e .
RUN export PYTHONPATH=`pwd`/../icefall:$PYTHONPATH
RUN echo "export PYTHONPATH=`pwd`/../icefall:\$PYTHONPATH" >> ~/.zshrc
RUN echo "export PYTHONPATH=`pwd`/../icefall:\$PYTHONPATH" >> ~/.bashrc
RUN cd ../..
# RUN source ~/.zshrc
# RUN source ~/.bashrc

RUN cp /opt/conda/lib/libpython3.10.so.1.0 /usr/lib/x86_64-linux-gnu/


WORKDIR /workspace/dependencies
RUN git clone https://github.com/lifeiteng/vall-e
WORKDIR /workspace/dependencies/vall-e
RUN pip install -e .

RUN pip install vocos gradio ipykernel
RUN pip install -U Ninja2

WORKDIR /workspace