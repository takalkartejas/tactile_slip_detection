FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Set environment variable to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other necessary libraries
RUN apt-get update && apt-get install -y \
    python3 
RUN apt-get install -y python3-pip
RUN apt-get install -y libgl1-mesa-glx 
RUN apt-get install -y libglib2.0-0 
RUN rm -rf /var/lib/apt/lists/*

# Set timezone to UTC (or your desired timezone)
RUN ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && dpkg-reconfigure -f noninteractive tzdata

#create working directory
WORKDIR /app


#Copy files form the local system
ADD /requirements.txt ./requirements.txt

#change the default shell to bash

SHELL ["/bin/bash", "-c"]


RUN pip install --upgrade pip
# RUN pip install tacto
RUN pip install -r requirements.txt
RUN pip install numpy



