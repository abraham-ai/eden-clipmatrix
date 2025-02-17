# set base image (host OS)
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

# remove this when they get a new cuda image out
# https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212771/9
RUN sh -c 'echo "APT { Get { AllowUnauthenticated \"1\"; }; };" > /etc/apt/apt.conf.d/99allow_unauth'
RUN apt -o Acquire::AllowInsecureRepositories=true -o Acquire::AllowDowngradeToInsecureRepositories=true update
RUN apt-get install -y curl wget
RUN apt-key del 7fa2af80
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/apt.conf.d/99allow_unauth cuda-keyring_1.0-1_all.deb
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC F60F4B3D7FA2AF80
RUN apt-get update && apt-get upgrade -y

# install dependencies
RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && apt-get -y update \
    && add-apt-repository universe
RUN apt-get -y update
RUN apt install -y libgl1-mesa-glx git wget zip libglib2.0-0 curl libsuitesparse-dev

# copy the content of the local src directory to the working directory
WORKDIR /usr/local/eden
COPY . .

# python libraries
RUN ./setup.sh

# command to run on container start
ENTRYPOINT [ "python3", "server.py" ]
