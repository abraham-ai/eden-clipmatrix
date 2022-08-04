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
RUN pip install --upgrade pip
RUN pip install --no-deps ftfy regex tqdm
RUN pip install largesteps scikit-image scikit-sparse libsuitesparse-dev scikit-learn

# install pytorch3d from source
RUN curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
RUN tar xzf 1.10.0.tar.gz
RUN CUB_HOME=/cub-1.10.0 pip install git+https://github.com/facebookresearch/pytorch3d.git@stable

# some more dependencies
RUN git clone https://github.com/openai/CLIP.git
RUN git clone https://github.com/rgl-epfl/large-steps-pytorch.git
RUN git clone https://github.com/lessw2020/Ranger21.git
RUN python -m pip install -e Ranger21/.

# files
RUN wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1CK2wrRqu94kPy3j4lm70UWaojrjgqO1F' -O model.obj

# command to run on container start
ENTRYPOINT [ "python3", "server.py" ]
