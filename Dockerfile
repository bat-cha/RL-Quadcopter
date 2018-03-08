FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
LABEL maintainer="bat-cha <baptiste.chatrain@gmail.com>"

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-setuptools \
        ros-kinetic-ros-base \
        python-rosinstall python-rosinstall-generator python-wstool build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app
VOLUME "/app"
WORKDIR "/app"
EXPOSE 11311

COPY requirements.txt /app/requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install -r /app/requirements.txt

RUN rosdep init && rosdep update 

# setup entrypoint
COPY ./ros_entrypoint.sh /

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
