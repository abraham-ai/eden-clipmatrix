import sys
sys.path.append("large-steps-pytorch/largesteps")

import os
#can slow down os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:100"

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np
import shutil
import torchvision


#@title Generator Options

TEXSIZE = 512#@param {type:'raw'}#size in pixels of square texture image - larger sizes are costly
HRL=1#@param {type:"slider", min:0, max:2, step:1} 
NClones=4#@param {type:"slider", min:1, max:10, step:1} 

train_num_views =3#@param {type:"slider", min:1, max:6, step:1} 


#if 0 use original mesh vertex resolution, if 1 or more - use mesh subdivision for more detail - more resource hungry
mesh_obj_name = "model.obj"#@param {type:'raw'}#a sample human mesh for quick testing - feel free to try any 3d mesh you have
# e.g. https://free3d.com/3d-models/ has many free resources

sideX=768#@param {type:'raw'}#size of image for final videos we will be saving

#prompts to use, we will sample randomly from them if multiple
headprompt = "head with 8 eyes "#@param {type:'string'}, ""
bodyprompt = "torso armored body of "#@param {type:'string'}, ""
legprompt = "legs and tail of "#@param {type:'string'}, ""

creatures = ["Mecha Warrior"] #@param {type:'raw'}
detail = ["detailed to the maximum","detailed to the maximum","rendered in Unreal Engine"]#@param {type:'raw'}, ""
styles = ["in sci fi Giger style "]#@param {type:'raw'}
#note: you can also use an image as style -- simply add the image file as prompt.
# e.g. download some "image.jpg" and set styles +=["image.jpg"]

#the larger you set these, the wilder the model will become and deform away from initial mesh
initGamma=5e-2#@param {type:'raw'}
initPV=0.01#@param {type:'raw'}

optisteps = 296 #@param {type:"slider", min:100, max:600, step:1} 
#how many gradient steps for optimization

extra_patch_clip =  1 #@param {type:"slider", min:1, max:9, step:1} 
clip_flip =  False  #@param {type:"boolean"}
# how many extra patches to crop in CLIP, in addition to 3d cameras - costly

fSobolevStrength = .2+4. #@param {type:'raw'}# the larger this is, the more regularized the mesh will be
fPenaltyDeformation = 1e-1#@param {type:'raw'}#the larger this is , the smoother the mesh and close to initial one
fPenaltyDeformation_e = 1e-2#@param {type:'raw'}
fPartRegularize = 1e-2#@param {type:'raw'} # for clones if more than 1

bWeightedEdges=False  #@param {type:"boolean"} 
#if True will use mesh edge distance for vertex Laplacian regularization
viewport_limit=  49#@param {type:"slider", min:0, max:60, step:1} 
# how many degrees to sample for rotation views

#regularize % of 3d mesh as part of total image - experimental
fSilhTarget = 0.24#@param {type:'raw'}
fSilhStrength = 4e1#@param {type:'raw'}

#some neural renderer settings
fShader_reg = 1e-2 #@param {type:'raw'}# with larger penalty, will be closed to underlying 3d model; with smaller penalty- let it be more wild
channels_neural_shader = 16 #@param {type:"slider", min:0, max:80, step:1} 
# the more channels, the more memmory hungry but fancier shading you get

useSymmetry = True #@param {type:"boolean"}
#use x-axis mesh template symmetry
useShadows = True #@param {type:"boolean"}
#use shadows as effect in rendering

##note: set these two options to false for better results if you are exporting mesh/texture to Blender
useNormalMaps = True #@param {type:"boolean"}
#use normal mapping for more detail
useNeuralShader = True #@param {type:"boolean"}
#use neural shading for more light effects

bSOBOLEV = True #@param {type:"boolean"}

create_video = True  #@param {type:"boolean"}
bsave_obj = True #@param {type:"boolean"}
save_texture = True #@param {type:"boolean"}
show_3d_interactive_runall = True #@param {type:"boolean"}

graphConv=True #@param {type:"boolean"}
nGraphLayers = 4 #@param {type:"slider", min:2, max:9, step:1} 

fgraphnet=0.6












#torch3d
need_pytorch3d=False
try:
    import pytorch3d
except ModuleNotFoundError:
    need_pytorch3d=True#works, needs restart however Nov. 11 2021    
    #!python3 -c "import torch;assert torch.__version__.startswith('1.6'), 'should be 1.6.x'" || pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html --upgrade 
    #!pip install pytorch3d

if need_pytorch3d:
    print ("needs pytorch",torch.__version__,torch.version.cuda)
    if False:#slow but works with latest COLAB, TODO switch to wheels once Pytorch3d fixed 
          os.system("pip install 'git+https://github.com/facebookresearch/pytorch3d.git")
    elif True:
        pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
        version_str="".join([f"py3{sys.version_info.minor}_cu",\
        torch.version.cuda.replace(".",""),f"_pyt{pyt_version_str}"])
        os.system("pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html")
    elif torch.__version__.startswith("1.9") and sys.platform.startswith("linux"):
        # We try to install PyTorch3D via a released wheel.
        version_str="".join([
            f"py3{sys.version_info.minor}_cu",
            torch.version.cuda.replace(".",""),
            f"_pyt{torch.__version__[0:5:2]}"
        ])
        os.system("pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html")
    else:
        # We try to install PyTorch3D from source.
        os.system("curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz")
        os.system("tar xzf 1.10.0.tar.gz")
        os.environ["CUB_HOME"] = os.getcwd() + "/cub-1.10.0"
        os.system("pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'")
import pytorch3d
print (pytorch3d.__version__)






from pytorch3d.io import load_objs_as_meshes, save_obj

from pytorch3d.loss import (
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex
)

from pytorch3d.io import save_obj



