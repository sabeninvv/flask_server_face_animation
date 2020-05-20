FROM ubuntu:latest
MAINTAINER Vladislav Sabenin 'vladislav.sabenin@app.coolrocket.com'
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get -y update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y python3-dev
RUN apt-get install -y build-essential
RUN apt-get install -y libglib2.0
RUN apt-get install -y ffmpeg
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
RUN pip3 install opencv-contrib-python-headless
RUN pip3 install https://download.pytorch.org/whl/cpu/torch-1.5.0%2Bcpu-cp38-cp38-linux_x86_64.whl
RUN pip3 install https://download.pytorch.org/whl/cpu/torchvision-0.6.0%2Bcpu-cp38-cp38-linux_x86_64.whl
RUN pip3 install face-alignment
CMD python3 main.py