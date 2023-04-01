FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 as base

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get install -y python3-pip 

RUN pip install torch==2.0.0
RUN pip install transformers==4.27.3
RUN pip install fairseq==0.12.2
RUN pip install onnx==1.13.1
RUN pip install onnxruntime==1.14.1
RUN pip install onnxsim==0.4.19

WORKDIR /
ADD convert.py /
ENTRYPOINT ["python3", "convert.py"]