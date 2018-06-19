
####################################################################################################
#
#   Tensorflow Object Detection API
#
#   Repo: https://github.com/tensorflow/models/tree/master/research/object_detection
#
####################################################################################################

#BUILD: 
# 	docker build -t tensorva .

#ENABLE XQUARTZ (on Mac) for remote display from docker container to local host / laptop
#https://sourabhbajaj.com/blog/2017/02/07/gui-applications-docker-mac/
# 	1) Open XQuartz
# 	2) Make sure Preferences > Security > Allow connections from network clients is Enabled/Checked
# 	3) Open up XQuarts terminal and type: xhost + <IP_of_local_host_or_laptop>

#RUN:
#	docker stop tensorva
#   docker rm tensorva
#	docker run -it -d -p 18080:8080 -p 19090:9090 -p 15555:5555 -p 14444:4444 --hostname tensorva -e DISPLAY=<IP_of_local_host_or_laptop>:0 -v /tmp/.X11-unix:/tmp/.X11-unix --name tensorva tensorva
# 	docker exec -it tensorva bash
# -or-
#   docker run -it <container> bash


FROM ubuntu:16.04
MAINTAINER Dan Zaratsian

WORKDIR /
COPY assets /assets

# Install OS packages and dependencies 
RUN apt-get update &&       \
        apt-get install -y  \
        build-essential     \
        cmake               \
        git                 \
        wget                \
        curl                \
        unzip               \
		net-tools           \
		vim                 \
        protobuf-compiler

# Install MiniConda and Python Packages
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda3.sh \
&& chmod +x /opt/miniconda3.sh          \
&& /opt/miniconda3.sh -b -p /miniconda3 \
&& rm /opt/miniconda3.sh                \
&& /miniconda3/bin/pip install          \
    tensorflow                          \
    jupyter                             \
    matplotlib                          \
    Cython                              \
    pillow                              \
    lxml

ENV PATH="/miniconda3/bin:$PATH" 

# Git clone Google Tensorflow Repo, install dependencies
RUN cd /opt                                                     \
&& git clone https://github.com/tensorflow/models.git           \
&& git clone https://github.com/cocodataset/cocoapi.git         \
&& cd cocoapi/PythonAPI                                         \
&& make                                                         \
&& cp -r pycocotools /opt/models/research/                      \
&& cd /tmp/                                                     \
&& curl -OL https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip \
&& unzip protoc-3.3.0-linux-x86_64.zip -d protoc3               \
&& mv protoc3/bin/* /usr/local/bin/                             \
&& mv protoc3/include/* /usr/local/include/                     \
&& cd /opt/models/research/                                     \
&& protoc object_detection/protos/*.proto --python_out=.

ENV PYTHONPATH="$PYTHONPATH:/opt/models/research:/opt/models/research/slim"




#ZEND
