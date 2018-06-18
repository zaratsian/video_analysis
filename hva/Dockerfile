
#BUILD: 
# 	docker build -t hva .

#ENABLE XQUARTZ (on Mac) for remote display from docker container to local host / laptop
#https://sourabhbajaj.com/blog/2017/02/07/gui-applications-docker-mac/
# 	1) Open XQuartz
# 	2) Make sure Preferences > Security > Allow connections from network clients is Enabled/Checked
# 	3) Open up XQuarts terminal and type: xhost + <IP_of_local_host_or_laptop>

#RUN:   
#	docker stop hva
#   docker rm hva
#	docker run -it -d -p 18080:8080 -p 19090:9090 -p 15555:5555 -p 14444:4444 --hostname hva -e DISPLAY=<IP_of_local_host_or_laptop>:0 -v /tmp/.X11-unix:/tmp/.X11-unix --name hva hva
# 	docker exec -it hva bash

FROM python:3.6
MAINTAINER Dan Zaratsian

RUN apt-get update && \
        apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
	net-tools \
	nano \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
	libgtk2.0-dev

# Install python packages
RUN pip install numpy==1.14.4 imutils==0.4.6 jupyter==1.0.0 matplotlib==2.2.2

WORKDIR /
COPY assets /assets

ENV OPENCV_VERSION="3.4.1"
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
&& unzip ${OPENCV_VERSION}.zip \
&& mkdir /opencv-${OPENCV_VERSION}/cmake_binary \
&& cd /opencv-${OPENCV_VERSION}/cmake_binary \
&& cmake -DBUILD_TIFF=ON \
  -DBUILD_opencv_java=OFF \
  -DWITH_CUDA=OFF \
  -DENABLE_AVX=ON \
  -DWITH_OPENGL=ON \
  -DWITH_OPENCL=ON \
  -DWITH_IPP=ON \
  -DWITH_TBB=ON \
  -DWITH_EIGEN=ON \
  -DWITH_V4L=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DCMAKE_INSTALL_PREFIX=$(python3.6 -c "import sys; print(sys.prefix)") \
  -DPYTHON_EXECUTABLE=$(which python3.6) \
  -DPYTHON_INCLUDE_DIR=$(python3.6 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
  -DPYTHON_PACKAGES_PATH=$(python3.6 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") .. \
&& make install \
&& rm /${OPENCV_VERSION}.zip \
&& rm -r /opencv-${OPENCV_VERSION}


#ZEND
