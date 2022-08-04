# python dependencies
pip install --upgrade pip
pip install --no-deps ftfy regex tqdm
pip install largesteps scikit-image scikit-sparse libsuitesparse-dev scikit-learn

# install pytorch3d from source
curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz
CUB_HOME=/cub-1.10.0 pip install git+https://github.com/facebookresearch/pytorch3d.git@stable

# ffmpeg
conda install ffmpeg

# some more dependencies
git clone https://github.com/openai/CLIP.git
git clone https://github.com/rgl-epfl/large-steps-pytorch.git
git clone https://github.com/lessw2020/Ranger21.git
python -m pip install -e Ranger21/.

# files
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1CK2wrRqu94kPy3j4lm70UWaojrjgqO1F' -O model.obj
