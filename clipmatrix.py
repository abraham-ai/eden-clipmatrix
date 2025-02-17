resourcePath = '.'

import os
from os import path

#Simple create paths taken with modifications from Datamosh's Batch VQGAN+CLIP notebook
def createPath(filepath):
    if path.exists(filepath) == False:
      os.makedirs(filepath)
      print(f'Made {filepath}')
    else:
      print(f'filepath {filepath} exists.')

initDirPath = f'{resourcePath}/init_obj'
createPath(initDirPath)
outDirPath = f'{resourcePath}/output'
createPath(outDirPath)
videoDirPath = f'{outDirPath}/video'
createPath(videoDirPath)
meshDirPath = f'{outDirPath}/mesh'
createPath(meshDirPath)

#videos saved there
vid_dir = videoDirPath




import os
import sys

#sys.path.append("large-steps-pytorch/largesteps")
sys.path.append("Ranger21/ranger21")

import torch
import torch.nn as nn
import torch.nn.functional as F
#import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np
import shutil
import torchvision

#@title Generator Options

TEXSIZE = 512#@param {type:'raw'}#size in pixels of square texture image - larger sizes are costly
HRL=1#@param {type:"slider", min:0, max:2, step:1} 
NClones=2#@param {type:"slider", min:1, max:10, step:1} 
UVclone=True#@param {type:"boolean"} 
#if True - use unique textures for each
fPartRegularize = 1e0#@param {type:'raw'} # for clones if more than 1

train_num_views =2#@param {type:"slider", min:1, max:6, step:1} 

#if 0 use original mesh vertex resolution, if 1 or more - use mesh subdivision for more detail - more resource hungry
mesh_obj_name = "model.obj"#@param {type:'raw'}#a sample human mesh for quick testing - feel free to try any 3d mesh you have
# e.g. https://free3d.com/3d-models/ has many free resources
sideX=768#@param {type:'raw'}#size of image for final videos we will be saving

#prompts to use, we will sample randomly from them if multiple
headprompt = "head of  "#@param {type:'string'}, ""
bodyprompt = "torso muscular body of "#@param {type:'string'}, ""
legprompt = "legs and tail of "#@param {type:'string'}, ""

#creatures = ["the red japanese robot ninja"] #@param {type:'raw'}

creatures = ["the steampunk robot detective"] #@param {type:'raw'}


detail = ["detailed to the maximum","detailed to the maximum","rendered in Unreal Engine"]#@param {type:'raw'}, ""
styles = ["in super realistic painting style"]#@param {type:'raw'}
#note: you can also use an image as style -- simply add the image file as prompt.
# e.g. download some "image.jpg" and set styles +=["image.jpg"]

#the larger you set these, the wilder the model will become and deform away from initial mesh
initGamma=2e-2#@param {type:'raw'}
initPV=0.01#@param {type:'raw'}

optisteps = 298 #@param {type:"slider", min:100, max:600, step:1} 
#how many gradient steps for optimization

extra_patch_clip =  1 #@param {type:"slider", min:1, max:9, step:1} 
clip_flip =  False  #@param {type:"boolean"}
# how many extra patches to crop in CLIP, in addition to 3d cameras - costly

fSobolevStrength = .2+2. #@param {type:'raw'}# the larger this is, the more regularized the mesh will be
fPenaltyDeformation = 1e1#@param {type:'raw'}#the larger this is , the smoother the mesh and close to initial one
fPenaltyDeformation_e = 1e1#@param {type:'raw'}

bWeightedEdges=False  #@param {type:"boolean"} 
#if True will use mesh edge distance for vertex Laplacian regularization
viewport_limit=  48#@param {type:"slider", min:0, max:90, step:1} 
# how many degrees to sample for rotation views

#regularize % of 3d mesh as part of total image - experimental
fSilhTarget = 0.24#@param {type:'raw'}
fSilhStrength = 4e1#@param {type:'raw'}

#some neural renderer settings
fShader_reg = 1e-2 #@param {type:'raw'}# with larger penalty, will be closed to underlying 3d model; with smaller penalty- let it be more wild
channels_neural_shader = 24 #@param {type:"slider", min:0, max:80, step:1} 
# the more channels, the more memmory hungry but fancier shading you get

useSymmetry = False #@param {type:"boolean"}
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
fgraphnet=0.3










# Util function for loading meshes
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


# add path for demo utils functions 
import sys
import os
sys.path.append(os.path.abspath(''))

from pytorch3d.io import load_objs_as_meshes
from PIL import Image
from pytorch3d.io import load_obj
from pytorch3d.ops import laplacian,norm_laplacian
from pytorch3d.transforms import RotateAxisAngle


# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Place a point light in front of the object. As mentioned above, the front of the cow is facing the -z direction. 
canonicL= PointLights(device=device, location=[[0,0,10]])# PointLights(device=device, location=[[0.0, 0.0, -3.0]]) #
lights = canonicL
R, T = look_at_view_transform(dist=1.9, elev=20, azim=0)
# We arbitrarily choose one particular view that will be used to visualize 
camera = OpenGLPerspectiveCameras(device=device, R=R,T=T) 
raster_settings = RasterizationSettings(
    image_size=sideX, 
    blur_radius=0.0, 
    faces_per_pixel=1, perspective_correct=True
)

# Create a phong renderer by composing a rasterizer and a shader. The textured 
# phong shader will interpolate the texture uv coordinates for each vertex, 
# sample from a texture image and apply the Phong lighting model
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=camera, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=camera,
        lights=lights
    )
)


# Data structures and functions for rendering
from pytorch3d.structures import Meshes
#from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib


from pytorch3d.loss import (
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    FoVOrthographicCameras,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    HardFlatShader,
    TexturesUV,
    TexturesVertex
)


from typing import Tuple, List
from typing import Optional, Dict, Union
from pytorch3d.renderer.utils import TensorProperties
from typing import Optional,Union
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer.mesh.shading import _apply_lighting
from pytorch3d.renderer.blending import softmax_rgb_blend
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.renderer.utils import TensorProperties
from pytorch3d.renderer import SoftPhongShader


class deform_oracle(object):
        def __init__(self):
          self.net=None
          self.last=None
          self.shaderI=None
        
          self.collision=0

        def calc(self,v_posed,pose_f=None,betas=None):
          if self.last is not None:
            return  v_posed+self.last

          if self.net is None:
            return 0
          return self.net(v_posed,pose_f)

ORACLE = deform_oracle()


from pytorch3d.ops.subdivide_meshes import SubdivideMeshes

def subdivide(m,HRL=HRL):
  if HRL==1:
    return SubdivideMeshes()(m)
  if HRL==2:
    def SubSub(m):
      s=SubdivideMeshes()
      return s(s(m))
    return SubSub(m)


#all tensors 3dim-except faces_iuvs
def repM(v10,f10,vt0,faces_uvs0):
  if NClones ==1:
    return v10,f10,vt0,faces_uvs0
  bv=[]
  bf=[]
  bvt=[]
  bft=[]

  nv=v10.shape[1]
  nvt=vt0.shape[1]
  N=NClones
  for i in range(N):
    off=0
    if i>0:
      off=torch.zeros(1,1,3).cuda().uniform_(-0.00002,0.00002)#just for sym
    bv.append(v10+off)#hack, just some offset first
    bf.append(f10+i*nv)
    vt0_=vt0*1
    if UVclone:
      vt0_[...,0]/=N#shrink
      vt0_[...,0]+= i/N#offset
    bvt.append(vt0_)#TODO other texture region
    bft.append(faces_uvs0+i*nvt)
    #plt.scatter(vt0_[0,:,0].cpu(),vt0_[0,:,1].cpu())

  #plt.show()
  return torch.cat(bv,1),torch.cat(bf,1),torch.cat(bvt,1),torch.cat(bft,0)



verts, faces, aux=load_obj(mesh_obj_name)

v10=verts.unsqueeze(0).cuda()
f10 = faces.verts_idx.unsqueeze(0).cuda()
vt=aux.verts_uvs.cuda().unsqueeze(0)
print("vt",vt.max(),vt.min())
faces_uvs = faces.textures_idx.cuda()
print (vt.shape,faces_uvs.shape,faces_uvs.max())

vt0 = vt*1
faces_uvs0 = faces_uvs*1
  
  #raise Exception

if HRL>0:##TODO add subdivide logic
        
    newm = subdivide(Meshes(v10,f10))
    orig_v_template=newm.verts_padded().squeeze()
    NVOrig0 = v10.shape[1]
    NVOrig = orig_v_template.shape[0]
    print ("NVorig",NVOrig,"orig",NVOrig0)
    
    f2=newm.faces_padded()
    print("v2",orig_v_template.shape)
    print ("faces",f2.shape,f2.max(),"old faces",f10.shape,f10.max())
    print (f2.max().item()+1)

    #for multiscale - indices of clones low res mesh
    ix_level0=[]
    for i in range(NClones):#add low level for each clone
      off=i*orig_v_template.shape[0]#verts in next cloned mesh, offset for initial level
      ix_level0+=list(range(off,off+NVOrig0))
    _,f10,_,_=repM(v10,f10,vt0,faces_uvs0)#only faces used in multiscale logic       
    
    newm = subdivide(Meshes(torch.cat([vt0,vt0[:,:,:1]*0],2),faces_uvs0.unsqueeze(0)))
    vt=newm.verts_padded()[...,:2]
    faces_uvs=newm.faces_padded().squeeze()

    ##repeat the cloning - -with high res mesh!
    orig_v_template,f2,vt,faces_uvs=repM(orig_v_template.unsqueeze(0),f2,vt,faces_uvs)
    orig_v_template=orig_v_template.squeeze()
else:
    #clone
    print (v10.shape,f10.shape,vt0.shape,faces_uvs0.shape)
    v10,f10,vt0,faces_uvs0=repM(v10,f10,vt0,faces_uvs0)
    print (v10.shape,f10.shape,vt0.shape,faces_uvs0.shape)   
    #other standard code 
    orig_v_template= v10.squeeze()
    NVOrig = orig_v_template.shape[0]
    f2=f10
    faces_uvs=faces_uvs0
    vt=vt0

#raise Exception


if False:
  fig = plot_scene({
        "text_to_add": {
            "mean optimized": m_check
        }
    })
  fig.show()


iHead= orig_v_template[:,1]>0.16
iTorso= (orig_v_template[:,1]>-0.4)*(orig_v_template[:,1]<0.16)*(orig_v_template[:,0].abs()<0.16)
iLeg= (orig_v_template[:,1]<-0.46)
iHand= (orig_v_template[:,0]>0.56)
iHandr= (orig_v_template[:,0]<-0.56)


if NVOrig< iHead.shape[0]:# and False:
  iHead[NVOrig:]=False
  iTorso[NVOrig:]=False
  iLeg[NVOrig:]=False
  iHand[NVOrig:]=False
  iHandr[NVOrig:]=False

# plt.scatter(orig_v_template[:,0].cpu(),orig_v_template[:,1].cpu(),c=iHead.cpu(),alpha=0.1)
# plt.show()
# plt.scatter(orig_v_template[:,0].cpu(),orig_v_template[:,2].cpu(),c=iHead.cpu(),alpha=0.1)
# plt.show()


# plt.scatter(orig_v_template[:,0].cpu(),orig_v_template[:,1].cpu(),c=iTorso.cpu(),alpha=0.1)
# plt.show()
# plt.scatter(orig_v_template[:,0].cpu(),orig_v_template[:,2].cpu(),c=iTorso.cpu(),alpha=0.1)
# plt.show()


# plt.scatter(orig_v_template[:,0].cpu(),orig_v_template[:,1].cpu(),c=iLeg.cpu(),alpha=0.1)
# plt.show()
# plt.scatter(orig_v_template[:,0].cpu(),orig_v_template[:,2].cpu(),c=iLeg.cpu(),alpha=0.1)
# plt.show()

# plt.scatter(orig_v_template[:,0].cpu(),orig_v_template[:,1].cpu(),c=iHand.cpu(),alpha=0.1)
# plt.show()
# plt.scatter(orig_v_template[:,0].cpu(),orig_v_template[:,1].cpu(),c=iHandr.cpu(),alpha=0.1)
# plt.show()
parts = [iHead,iTorso,iLeg,iHand,iHandr]


#@title text prompts logic - aligned with parts definitions
def sample(a):
    return a[np.random.randint(len(a))].replace('\n','')
handprompt="hand with mechanic fingers of "   
def getTexts():
    acreature= sample(creatures)
    astyle= sample(styles)    
    det =  detail[np.random.randint(len(detail))]
    text_0 =   acreature + " " + astyle + " " + det
    text_0bw = acreature+ " " + astyle

    text_1  = str(headprompt)+" "+text_0
    text_1bw = str(headprompt)+" "+text_0bw

    text_2  = str(bodyprompt)+" "+text_0
    text_2bw = str(bodyprompt)+" "+text_0bw 

    text_3 = str(legprompt) + text_0 
    text_3bw =text_3

    text_4 = str(handprompt) + text_0 
    text_4bw =text_4

    descriptor=text_0
    
    allt=[text_0,text_0bw,text_1,text_1bw,text_2,text_2bw,text_3,text_3bw,text_4,text_4bw]
    allt += allt[-2:]
    return allt,descriptor



iKnight=orig_v_template[:,1]>0.16
iKnight[NVOrig:]=False
iKnight[:NVOrig]=True
iOrc=orig_v_template[:,1]>0.16
iOrc[NVOrig:]=True
iOrc[:NVOrig]=False

iHeadK= orig_v_template[:,1]>0.16
iHeadK[NVOrig:]=False
iHeadO= orig_v_template[:,1]>0.16
iHeadO[:NVOrig]=False

#hak - use 2 whole figures
parts = [iKnight,iOrc,iHeadK,iHeadO]

def getTexts():
    acreature= "knight fighting an orc"
    astyle= sample(styles)    
    det =  detail[np.random.randint(len(detail))]
    text_0 =   acreature + " " + astyle + " " + det
    text_0bw = acreature+ " " + astyle

    text_1  = "armored knight" + " " + astyle + " " + det
    text_1bw = text_1

    text_2  = "orc barbarian" + " " + astyle + " " + det
    text_2bw = text_2

    text_3  = "head with eyes and mouth of armored knight" + " " + astyle + " " + det
    text_3bw = text_3

    text_4  = "head with eyes and mouth of orc barbarian" + " " + astyle + " " + det
    text_4bw = text_4

    descriptor=text_0
    
    allt=[text_0,text_0bw,text_1,text_1bw,text_2,text_2bw,text_3,text_3bw,text_4,text_4bw]
    return allt,descriptor


print (f2.shape)
v2= orig_v_template.unsqueeze(0)#Tshape
#v2=output.vertices.detach()#.cpu()#.numpy().squeeze()
print (v2.device,v2.shape,v2.dtype,f2.device,f2.shape,f2.dtype)
print (f2.max(),f2.min(),f2[0,0])

with torch.no_grad():
  mesh3 = Meshes(v2*3-0.3,f2)
  mesh3.textures = TexturesVertex(verts_features=v2*0) #
  #mesh3.textures =texture#

  l_e_orig = mesh_edge_loss(mesh3) 
  l_l_orig =  mesh_laplacian_smoothing(mesh3)
  l_n_orig =  mesh_normal_consistency(mesh3)

with torch.no_grad():
  lights = canonicL#PointLights(device=device, location=[[0.0, 0.0, -3.0]])#hack to reset to normal light
  images = renderer(mesh3, lights=lights)
  print (images.shape,images.max(),images.min())#directly 0,1 -- ok
  #plt.figure(figsize=(20,20))
  #plt.imshow(images[0,:,:,:3].cpu())
  #plt.show()

#raise Exception


def mesh_edge_len(meshes):
    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    N = len(meshes)
    edges_packed = meshes.edges_packed()  # (sum(E_n), 3)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)

    edge_to_mesh_idx = meshes.edges_packed_to_mesh_idx()  # (sum(E_n), )
    num_edges_per_mesh = meshes.num_edges_per_mesh()  # N

    verts_edges = verts_packed[edges_packed]
    v0, v1 = verts_edges.unbind(1)
    l=(v0 - v1).norm(dim=1, p=2) 
    return l
with torch.no_grad():
  edge_len = mesh_edge_len(mesh3)#initial mesh
print ("initial edges",edge_len.shape)

wel = edge_len*0+1
# plt.scatter(edge_len.cpu(),wel.cpu())
# plt.show()


usePerspective=False
def getCamP(R,T):
    if usePerspective:
        return FoVPerspectiveCameras(device=device, R=R, T=T,fov=30)##fov like zoom? calibrate negates

    return FoVOrthographicCameras(device=device, R=R, T=T)


fv=50+40#larger fov and close to camera -- large distort
cdist=2.+1#if small -- more distort

num_views = 360#used in target cameras only

easc=15
elev = torch.linspace(-easc,+easc, num_views//2)
elev=torch.cat([elev,torch.flip(elev,[0])])

TaT = 0#hmm, only visible when having different angle than sun?
asc=0.001#no camera change gkobally - mesh will be rotated instead
azim = torch.linspace(TaT-asc,TaT+asc, num_views//2)
if False:
    azim=torch.cat([azim,torch.flip(azim,[0])])
else:
    azim=torch.cat([azim,azim])
R2, T2 = look_at_view_transform(dist=cdist, elev=elev, azim=azim)

#cameras =FoVPerspectiveCameras(device=device, R=R2, T=T2,fov=fv)##pespective? orthotraphic finer control with perspective, e.g. translation in ortho weird
num_views = train_num_views#4#8
   
#random  views
def sampleRC(num_views=num_views):
  if True:#sample in range
    elev = torch.zeros(num_views).uniform_(-easc,+easc)
    #asc=180#overwrite---
    azim = torch.zeros(num_views).uniform_(TaT-asc,TaT+asc)#try narrow angle
  else:
    elev = torch.linspace(16,17, num_views)
    azim = torch.linspace(-360-40,-360+40, num_views)
  R2, T2 = look_at_view_transform(dist=cdist, elev=elev, azim=azim)
  cameras =getCamP(R=R2, T=T2)
  return cameras


def norm_laplacian2(verts_packed,edges_packed):
    L= norm_laplacian(verts_packed,edges_packed)
    V=L.shape[0]
    # Add the diagonal indices
    vals = torch.sparse.sum(L, dim=0).to_dense()
    indices = torch.arange(V, device='cuda')
    idx = torch.stack([indices, indices], dim=0)
    L = torch.sparse.FloatTensor(idx, vals, (V, V)) - L
    return L


verts_packed = orig_v_template.squeeze()#mesh3.verts_packed()  # (sum(V_n), 3)
edges_packed = mesh3.edges_packed()  
from largesteps.solvers import ConjugateGradientSolver,solve,CholeskySolver
if bSOBOLEV:    
    #from solvers import *
    #I0 = torch.ones(verts_packed.shape[0])
    #I = torch.diag(I0).to_sparse().cuda()
    V=verts_packed.shape[0]
    idx=torch.arange(V).cuda()
    idx = torch.stack([idx,idx], dim=0)
    I = torch.sparse.FloatTensor(idx, torch.ones(V).cuda(), (V, V))
    if not bWeightedEdges:#normed with degree
        L = -laplacian(verts_packed,edges_packed)#so diagonal is 1
    else: #inv distance - so small polys smoother
        L= 1e-2*norm_laplacian2(verts_packed,edges_packed)
    M=I+fSobolevStrength*L#
    solver=ConjugateGradientSolver(M.coalesce())#
    #solver = CholeskySolver(M.coalesce())
    def applyPreconditioner(x):
      return solve(solver,x)



def saveI(img,i):
  img=np.uint8(img*255)
  imageio.imwrite("frames2/"+str(i) + '.jpg',img,quality=95)

def saveImageSet(mesh3,images=[],text_to_add= "just a T"):
  start_offset= len(os.listdir("frames2/"))#in case adding to other frames
  print ("start offset",start_offset)
  images=torch.cat(images)
  for i in range(images.shape[0]):
    saveI(images[i,:,:,:3].cpu().numpy(),i+start_offset)



#Import CLIP and load the model
from CLIP import clip
print (clip.available_models())
clip_preprocess=  torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14']
if False:#standard openai
  perceptor, preprocess = clip.load('ViT-L/14', jit=False)#32 initially
  perceptor.eval().requires_grad_(False);
  clip.available_models()
elif True:#combine 2 models
  perceptor0,_ = clip.load('ViT-B/16', jit=False)#32 initially
  perceptor1,_ = clip.load('ViT-B/32', jit=False)#any difference?##arghh, uses other size!!! -- 288!
  perceptor0.eval().requires_grad_(False)
  perceptor1.eval().requires_grad_(False)
    
  print (perceptor0.visual.input_resolution )
  print (perceptor1.visual.input_resolution )
 
  class CompoClip(object):
    def __init__(self):
        self.p0=perceptor0
        self.p1=perceptor1
        
    def encode_text(self,t):
        return torch.cat([self.p0.encode_text(t),self.p1.encode_text(t)],-1)
          
    def encode_image(self,t):
        return torch.cat([self.p0.encode_image(t),self.p1.encode_image(t)],-1)
          
  perceptor=CompoClip()


pbatch_size=1

#replace the B H W C=4 channels and pass to CLIP
def procClip(x):
  return x.permute(0,3,1,2)[:,:3]

def soo(sideX):
  a=0.84
  size = int((      a+np.random.rand()*(1-a)  )*sideX)
  offsetx = torch.randint(0, int(sideX - size), ())
  offsety = torch.randint(0, int(sideX - size), ())
  return offsetx,offsety,size 

#gives crops patches proportional to size
def getP(cutn,sideX):
  p_s = []#just patch crops
  for ch in range(cutn):
      offsetx,offsety,size =soo(sideX)
      p_s.append((offsetx,offsety,size))
  return p_s

#can work with 1 image only
def patch(into,cutn=32):
  #up_noise=0.1
  scaler=1
  patch = getP(cutn,into.shape[2])
  p_s=[]
  for ch in range(cutn):
      offsetx,offsety,size = patch[ch]
      apper = into[:, :, offsetx:offsetx + size, offsety:offsety + size]
      apper = torch.nn.functional.interpolate(apper, (int(224*scaler), int(224*scaler)), mode='bilinear', align_corners=True)
      p_s.append(apper)

  into = torch.cat(p_s, 0)
  return into

#loss for text embedding t,t2 into image into; orig gives optionally mask where all 1s in three channels means to ignore!
def lossClip(into,t,t2=None,orig=None,use_patch=False):
  if orig is None:
    orig =into
  if extra_patch_clip >=1 and (use_patch or into.shape[2]!=224):
    into1=patch(into,cutn=extra_patch_clip)
    into=into1
  else:
    into =  torch.nn.functional.interpolate(into, (int(224), int(224)), mode='bilinear', align_corners=True)
  if np.random.rand()<0.01:# debug sanity
      print ("clip patch cuts",into.shape,"texts",len(t))
    
  into = clip_preprocess(into)
  if clip_flip:
    into = torch.cat([into,torch.flip(into,dims=[3])])  
  #no vert into = torch.cat([into,torch.flip(into,dims=[2])])  
  #print ("inti",into.shape)
  iii = perceptor.encode_image(into)
  l1=  0
  for te in t:
    v=torch.cosine_similarity(te, iii, -1)
    #v=v.mean()
    v=0.5*(v.max()+v.mean())#new experiment, only best view activates!!
    l1 += -10*v/len(t)
  if t2 is not None:
    for part in t2:
      l1= l1+  1*torch.cosine_similarity(part, iii, -1).mean()/len(t2)
  return l1



import pickle
if useSymmetry:  #try symmetry of mesh
  try:
    ixs=pickle.load(open('symVerts%d.dat'%(orig_v_template.shape[0]),'rb'))
    assert(len(ixs)==orig_v_template.shape[0])
  except:
      mir = orig_v_template*1
      mir[:,0]*=-1#only x coordinate!
      print (mir.shape)
      out=[]
      ixs=[]
      for i in range(orig_v_template.shape[0]):
        d = orig_v_template[i:i+1]-mir
        dd= (d**2).sum(1)
        m=dd.min()
        out.append(m.item())
        ixs.append(np.argmin(dd.cpu()))#closest mirror to vertex i
      out=np.array(out)
      pickle.dump(ixs,open('symVerts%d.dat'%(orig_v_template.shape[0]),'wb'))
      print ("SYM verts",out.mean(),out.min(),out.max())
  def sym(v):
    v2 = v[ixs,:]#for each vertex gives position of its mirror
    m = torch.cat([v2[:,:1]*-1,v2[:,1:]],1)#should have grad, project mirror across x axis
    return (m+v)/2

  test=sym(orig_v_template)
else:
  def sym(v):
    return v#dummy


#emebedding of mesh - -takes 1 minute, disable if not using
LAP_EMB = True
blockGC = False#if  True simple bias stuff!!
NE=120#50#try 50 or 150
bUseClust = False
loaded=False

if LAP_EMB:
  try:
      lname='meshEmb_L%d_%d.pt'%(HRL,NE+int(bUseClust)*22)
      plname='%s/'%(resourcePath) + lname
      mesh_embed= torch.load(plname)
      print ("loaded",mesh_embed.shape)
      mesh_embed=mesh_embed.cuda()
      
      NE = mesh_embed.shape[1]
      assert (mesh_embed.shape[0]==orig_v_template.shape[0])
      print ("loading worked")
      loaded=True
      #assert (mesh_embed.shape[1]==NE)
  except Exception as e:
      print ("could not load",e,orig_v_template.shape)
      print (e)
      #recalc
      #raise Exception
else:
  mesh_embed=1*orig_v_template
  NE =3

if LAP_EMB and not loaded:
  useSelf=True
  print ("use just one mesh clone if identical, no need for full embedding")
  f_02 = f2[:,:f2.shape[1]//NClones]
  s=f_02.max().item()+1
  print (f2.shape,f_02.shape,s,orig_v_template.shape)
  from scipy.sparse import lil_matrix
  A = lil_matrix((s,s))
  print ("A array",A.shape,A.dtype)

  vals=[]
  for i in range(f_02.shape[1]):
                  for j in range(3):
                      for jj in range(3):
                          if j !=jj:#self similarity or not
                              vals.append(1)
                              A[f_02[0,i,j].cpu(),f_02[0,i,jj].cpu()]=vals[-1]
                      if useSelf:
                          A[f_02[0,i,j].cpu(),f_02[0,i,j].cpu()]=0.0001 
  #plt.plot(np.sort(np.array(vals[::10])))
  #plt.title("adjacency vals")
  #plt.show()

  from sklearn.manifold import SpectralEmbedding
  tsne = SpectralEmbedding(NE//NClones,affinity ='precomputed')
  assert(NE%NClones ==0)
  NE0=NE//NClones
  zx=torch.FloatTensor(tsne.fit_transform(A)).cuda()#
  mesh_embed= torch.zeros(zx.shape[0]*NClones,NE).cuda()#zx
  for i in range(NClones):#copy blocks
    mesh_embed[i*zx.shape[0]:(i+1)*zx.shape[0],i*NE0:(i+1)*NE0]=zx
  del A



if LAP_EMB and not loaded:
    torch.save(mesh_embed.cpu(),plname)
    
NE=NE+3
mesh_embed= torch.cat([mesh_embed,orig_v_template],1)


mesh_embed-=mesh_embed.mean(0).unsqueeze(0)
mesh_embed /= mesh_embed.std(0).unsqueeze(0)


# if LAP_EMB:
#   plt.figure(figsize=(27,27))
#   for z in range(9):
#     plt.subplot(3,3,z+1)
#     plt.scatter(orig_v_template.cpu()[::4,0],orig_v_template.cpu()[::4,1],c=mesh_embed[::4,z].cpu(),s=3,alpha=0.4)
#     plt.axis('off')
#   plt.show()



from pytorch3d.ops.graph_conv import GraphConv,gather_scatter

class GraphConv(nn.Module):
    """A single graph convolution layer. with symmtric normalize"""
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        init: str = "normal",
        directed: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.directed = directed
        self.w0 = nn.Linear(input_dim, output_dim)
        self.w1 = nn.Linear(input_dim, output_dim)

        if init == "normal":
            nn.init.normal_(self.w0.weight, mean=0, std=0.01)
            nn.init.normal_(self.w1.weight, mean=0, std=0.01)
            # pyre-fixme[16]: Optional type has no attribute `data`.
            self.w0.bias.data.zero_()
            self.w1.bias.data.zero_()
        elif init == "zero":
            self.w0.weight.data.zero_()
            self.w1.weight.data.zero_()
        else:
            raise ValueError('Invalid GraphConv initialization "%s"' % init)

    def forward(self, verts, edges,deg):
        if verts.is_cuda != edges.is_cuda:
            raise ValueError("verts and edges tensors must be on the same device.")
        if verts.shape[0] == 0:
            # empty graph.
            return verts.new_zeros((0, self.output_dim)) * verts.sum()

        verts_w0 = self.w0(verts)  # (V, output_dim)
        verts_w1 = self.w1(verts*deg)  # (V, output_dim)

        if torch.cuda.is_available() and verts.is_cuda and edges.is_cuda:
            neighbor_sums = gather_scatter(verts_w1, edges, self.directed)
        else:
            neighbor_sums = gather_scatter_python(
                verts_w1, edges, self.directed
            )  # (V, output_dim)

        # Add neighbor features to each vertex's features.
        out = verts_w0 + deg*neighbor_sums
        return out
    
def deg(V,edges):
    e0, e1 = edges.unbind(1)
    idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)
    ones = torch.ones(idx.shape[1], dtype=torch.float32).cuda()
    A = torch.sparse.FloatTensor(idx, ones, (V, V))
    # the sum of i-th row of A gives the degree of the i-th vertex
    deg = torch.sparse.sum(A, dim=1).to_dense()
    return deg

#multiply by diagonal matrix
def ngc(s,g,x):
          #x=torch.cat([x,x.mean(0,keepdim=True)+x.detach()*0],-1)#pooling like!
          return g(x,s.edges,s.deg)



class PlainDeform(torch.nn.Module):
  def __init__(self,orig_v,extradim=0,gcdepth=None,nZ=50*3,useClip=0,initGamma=5e-2,initPV=1):
    super().__init__()
    #hack: *4 when using plain initially
    self.w = nn.Parameter(initPV*(orig_v*0).uniform_(-0.04,0.04))    #pure bias
    
    self.w1 = nn.Parameter(torch.zeros(NE,3).uniform_(-0.005,0.005))#linear on spectral features#this is too wild
    
    self.mlp = nn.Sequential(nn.Linear(NE,nZ),nn.ReLU(True),nn.Linear(nZ,nZ),nn.ReLU(True),nn.Linear(nZ,3))
    self.gamma0=nn.Parameter(torch.zeros(1)+1)
    self.gamma=nn.Parameter(torch.zeros(1)+initGamma)
    self.bSYM=useSymmetry

    if graphConv:
        z=32*1
        cin=3+NE
        self.nlin=nn.LeakyReLU(0.2)
        self.first=GraphConv(cin*1,z)
        self.edges = mesh3.edges_packed()
        self.deg=deg(orig_v_template.shape[0],self.edges).unsqueeze(1)
        self.deg = 1/torch.sqrt(self.deg)
        print ("edges",self.edges.shape,self.deg.shape)
        self.gc = nn.ModuleList()
        for i in range(nGraphLayers):
              self.gc.append(GraphConv(z*1,z))#was +6
        self.gc.append(GraphConv(z*1,3))
    
  def forward(self,v_posed):
    out=self.gamma0*self.w+ self.gamma*self.mlp(mesh_embed)+self.gamma*mesh_embed@self.w1    
    if graphConv:
        if self.bSYM:
          out=sym(out)
        z=ngc(self,self.first,torch.cat([mesh_embed,out],1))    
        br=0
        for g in self.gc:
          oldz=z

          z=self.nlin(z)
          if True and br !=len(self.gc)-1:
              z=oldz+fgraphnet*ngc(self,g,z)
          else:
              z=ngc(self,g,z)
          br +=1
        out=out+fgraphnet*z 

    if bSOBOLEV:#u reparametrize#ahaa try also full code, with the v_shaped; this is the from_differential command in the EPFL paper TODO solved of EPFL
      out = applyPreconditioner(out)
    if self.bSYM:
      out=sym(out)##          
    out=out+v_posed[:,:3]
    self.last=out-v_posed[:,:3]#orig_v_template#
    return out



def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        #nn.InstanceNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   

class UNet(nn.Module):
    def __init__(self, n_class,c_in=3,c=88):
        super().__init__()
        self.c=c

        self.dconv_down1 = double_conv(c_in, c)
        self.dconv_down2 = double_conv(c, 2*c)
        self.dconv_down3 = double_conv(2*c, 4*c)
        self.dconv_down4 = double_conv(4*c, 8*c)        

        self.maxpool = lambda x:F.interpolate(x, scale_factor=0.5, mode='bilinear')#nn.AvgPool2d(2)#nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(4*c + 8*c, 4*c)
        self.dconv_up2 = double_conv(2*c + 4*c, 2*c)
        self.dconv_up1 = double_conv(c + 2*c, c)
        self.conv_last = nn.Conv2d(c+c_in, 3, 1)

    #z is noise -- what scale?
    def forward(self, x_,zraw=None):
        conv1 = self.dconv_down1(x_)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        out = self.conv_last(torch.cat([x,x_],1))#new Dec. 2 -- also add last
        return out


#full grid structure
TEXSIZEw=TEXSIZE
if UVclone:
  TEXSIZEw*=NClones
def getImageL():
  ix=[]
  v=[]
  V=TEXSIZE*TEXSIZEw
  sk=0
  for h in range(TEXSIZE-1):
    for w in range(TEXSIZEw-1):
      
      i=h*TEXSIZEw + w#curr index

      j=i+1#hor neighbour
      
      ix.append([i,j])
      v.append(-1.0)
      ix.append([j,i])
      v.append(-1.0)
      ix.append([j,j])
      v.append(1.0)
      ix.append([i,i])
      v.append(1.0)

      j=i+TEXSIZEw#ver neighbour

      ix.append([i,j])
      v.append(-1.0)
      ix.append([j,i])
      v.append(-1.0)
      ix.append([j,j])
      v.append(1.0)
      ix.append([i,i])
      v.append(1.0)
  
  ix= np.array(ix).T#so 2 x N
  print ("skipped",sk,"did indices in image laplacian",ix.shape)##non-full -- still solve?
  #8x entries added per valid pixel; symmetry and back - so 4; diagonals i,j and ij,ji
  v=np.array(v)
  L= torch.sparse_coo_tensor(ix, v, (V,V)).coalesce().cuda()

  ix=np.arange(V)
  ix=np.int32(np.array([ix,ix]))#so 2xN
  v=np.ones(V)
  print (ix.shape,v.shape)
  I= torch.sparse_coo_tensor(ix, v, (V,V)).coalesce().cuda()
  return L,I
solveTex=True# and not UVclone
if solveTex:
  try:
      IL,II = torch.load('textureLaplacian%d.dat'%(TEXSIZEw))
      print ("loaded texture Laplacian")
  except:
      print ("calculating texture laplacian")
      IL,II=getImageL()
      torch.save((IL,II ),'textureLaplacian%d.dat'%(TEXSIZEw))
  print (IL.shape,II.shape)
  IM=II+(.8+0)*IL
  solverI = CholeskySolver(IM.coalesce().float())

  IM=II+(.8)*IL
  solverIN = CholeskySolver(IM.coalesce().float())


texunet=False
class TextureL(nn.Module):
    def __init__(self,c=6):
      super().__init__()
      tex=torch.zeros(1,TEXSIZE,TEXSIZEw,3).uniform_(-3,3)   # .uniform_(0.25,0.75)
      self.tex = nn.Parameter(tex)
      texn=torch.zeros(1,TEXSIZE,TEXSIZEw,3).uniform_(-0.2,0.2)
      self.texn = nn.Parameter(texn)

      if texunet:
          self.unet0= nn.Conv2d(3,3,7,1,3)
          self.unet1= nn.Conv2d(3,3,7,1,3)
     
    def forward(self,x=None,z=None):
      tex=torch.sigmoid(self.tex)
      texn=self.texn
        
      if solveTex:
        tex_ = solve(solverI,tex.contiguous().view(-1,3))
        tex=tex_.view(tex.shape)
        texn_ = solve(solverIN,texn.contiguous().view(-1,3))
        texn=texn_.view(texn.shape)

        if texunet:
          tex=tex+0.15*self.unet0(tex.permute(0,3,1,2)).permute(0,2,3,1)
          texn=texn+0.15*self.unet1(texn.permute(0,3,1,2)).permute(0,2,3,1)
        
      return tex,texn
     


from pytorch3d.renderer.utils import TensorProperties
from typing import Optional,Union
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer.mesh.shading import _apply_lighting
from pytorch3d.renderer.blending import softmax_rgb_blend


R_, T_ = look_at_view_transform(dist=5, elev=45, azim=0)
#if 0 - 0,0,10 light
#if 45 -> 0,1,1 light
canCam =getCamP(R=R_, T=T_)

def fragmentD(fragments):
  d=fragments.zbuf#1,H,W,fpp
  #use defaults, make to 1,H,W,fpp,3
  df= softmax_rgb_blend(d.unsqueeze(-1).repeat(1,1,1,1,3), fragments, BlendParams(), znear=1, zfar=100)
  #if False:
    #plt.imshow(df.detach().cpu()[0,:,:,0])#weird, back is 1, front is 5
    #plt.colorbar()
    #plt.show()
  return df[...,:1]#first channel is averaged depth

class ShaderWithDepth(nn.Module):
    def __init__(self):
        super().__init__()        
    def forward(self, meshes_world,size=None,**kwargs) -> torch.Tensor:
        raster_settings_shadow = RasterizationSettings(
            image_size=size, 
            blur_radius=np.log(1. / 1e-4 - 1.)*1e-5,
            faces_per_pixel=4, perspective_correct=usePerspective and False,max_faces_per_bin=orig_v_template.shape[0])#
        rasterizer = MeshRasterizer(cameras=canCam, raster_settings=raster_settings_shadow)     
        fragments = rasterizer(meshes_world)##TODO pass raster_settings to overwrite
        return fragmentD(fragments)#default blend params
        return fragments.zbuf#TODO any way to smooth?
    
depthShader = ShaderWithDepth()


NP=25
a=[]
for i in [-2,-1,0,1,2]:
  for j in [-2,-1,0,1,2]:
    a.append(torch.FloatTensor([i,j]).unsqueeze(0))
off = torch.cat(a)
print (off)
off = off.cuda().view(NP,1,1,1,2)/sideX
#off = torch.zeros(NP,1,1,1,2)
#off=off.cuda().uniform_(-1,1)/500#poisson disc

def calcShadows(meshes,texels,fragments):
     verts = meshes.verts_packed()  # (V, 3)
     faces = meshes.faces_packed()  # (F, 3)
    
     adjust = 1.5/(verts[:,1].max()-verts[:,1].min())#so how streched
     with torch.no_grad():
            depth = (depthShader(meshes,size=texels.shape[1])-1)/99.0
            #plt.figure(figsize=(14,14))
            #plt.imshow(depth[0,:,:,0].cpu())
            #plt.show()
            xyd = canCam.transform_points(verts)
            xy=xyd[...,:2]*-1#-1 1 ok, but any flipping required?
            d=xyd[...,2:3].view(-1)#depth per vertex, of mesh when seen from light, including occluded backsides
            shadow_coords =0
            for p in range(NP):
                zbuf= F.grid_sample(depth.permute(0,3,1,2),xy.view(depth.shape[0],-1,1,2)+off[p])#shadow map values at location
                #print ("zbuf",zbuf.shape,"d",d.shape)#zbuf torch.Size([2, 1, 41853, 1]) d torch.Size([83706])
                zbuf = zbuf.view(-1)
                delta=zbuf-d#so 0 when exactly lit, when negative . some occlusio
                delta*=adjust#so standard scale
                delta[delta >-6e-4]=0#where slightly off
                fdelta=delta.view(-1,1)[faces]
                
                shadow_coords_ = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, fdelta)
                shadow_coords_ =  torch.exp(3e3*(shadow_coords_))#smoother in low res?   

                shadow_coords +=shadow_coords_/NP 
                #shadow_coords=torch.clamp(shadow_coords,0,1)
                
            if np.random.rand() <0.1:
                print ("rendered shadow",shadow_coords.mean())
            return shadow_coords


regEye=torch.eye(2).unsqueeze(0).cuda()
def getTanSpace(mesh,vertex3,normals):
    out=[]
    faces_uvs=mesh.textures.faces_uvs_padded()[0].squeeze()##problem -- if other texture vertex textzre... TODO use just plain UVs!
    f2=mesh.faces_padded()[0]#so that proper crops of parts work
    for n in range(vertex3.shape[0]*0+1):
        E1= vertex3[n,f2[:,0]]-vertex3[n,f2[:,1]]
        E2= vertex3[n,f2[:,0]]-vertex3[n,f2[:,2]]#1 Nfaces 3     
        E12=torch.cat([E1.unsqueeze(1),E2.unsqueeze(1)],1)#concat, 1 Nfaces x 3x2
        
        dUV1= vt[0,faces_uvs[:,0]]-vt[0,faces_uvs[:,1]]#2d
        dUV2= vt[0,faces_uvs[:,0]]-vt[0,faces_uvs[:,2]]
        dUV12= torch.cat([dUV1.unsqueeze(1),dUV2.unsqueeze(1)],1)# concat 1 Nfaces x2 x2        
        I=torch.inverse(dUV12+1e-10*regEye)
        
        TB = I.bmm(E12)
        TB=F.normalize(TB,dim=-1)#make sure unit vectors!        
        N= normals[n]# from normals        
        NTB= torch.cat([N.unsqueeze(1),TB],1)#so all together, should be totalfaces x 3x3 -- how to transpose?
        out.append(NTB.permute(0,2,1))        
    return out#torch.cat(out)



#calculates tangent space normals
def sample_texturesN(fragments,mesh,texels_nm0):
    with torch.no_grad():
      NTB=getTanSpace(mesh,mesh.verts_padded(),mesh.faces_normals_padded())    
    out=[]
    X=texels_nm0##can we calculate first the normals per pixel in some other space? or do in HW space as usual
    
    #if X is usual texture, here a normal map -- move to tangent space for each
    if True:
      for n in range(len(T)):#per batch instance
          rot=NTB[n*0].view(-1,3,3)#so rotation matrix per face --tangent space        
          rotHW = rot[fragments.pix_to_face[n:n+1]]#1HWKx3x3   --fragment for one batch only     
          x=X[n:n+1]#for 1 instance , 1KHW3
          #print ("before rotation",x.shape,rotHW.shape,rot.shape)#before rotation torch.Size([1, 768, 768, 5, 3]) torch.Size([1, 768, 768, 5, 3, 3])
          tangent_x=torch.matmul(rotHW.view(-1,3,3),x.contiguous().view(-1,3,1))##will it work; is it correct directopn? does not change size
          out.append(tangent_x.view(x.shape))
      out=torch.cat(out)
      return out#NHWKC
    else:#assume all the same -- weird why such errors
      rot=NTB[0]
      rotHW = rot[fragments.pix_to_face]
      #print ("before rotation",X.shape,rotHW.shape,rot.shape)
      tangent_x=torch.bmm(rotHW.view(-1,3,3),X.contiguous().view(-1,3,1))
      return tangent_x.view(X.shape)

#combines tangent space nornamls and usual normals
#per fragment!
def phong_shading_nm(meshes, fragments, lights, cameras, materials, texels,texels_nm) -> torch.Tensor:    
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)##weird, why not face normals??
    if True:#canonical pytorch3d code; use interplation of vertex normals
      faces_normals = vertex_normals[faces]
      faces_normals=interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, faces_normals)
    else:
      faces_normals = meshes.faces_normals_packed() 
      faces_normals = faces_normals[fragments.pix_to_face]   
    faces_verts = verts[faces]    
    pixel_coords = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, faces_verts)    
    #normal map as residual to usual pixel normals! -- may need to reshape
    pixel_normals = texels_nm+faces_normals

    if useShadows and texels.shape[1]>224:#so proper texture, not mono, and also not small partial clip HACK
        shadow_coords=calcShadows(meshes,texels,fragments)
    else:
        shadow_coords=1
 
    ambient, diffuse, specular = _apply_lighting(
        pixel_coords, pixel_normals, lights, cameras, materials
    )
    colors = (ambient + shadow_coords*diffuse) * texels +shadow_coords*specular
    
    if ORACLE.shaderI is not None:
        return colors,F.normalize(pixel_normals,dim=-1)    
    return colors,None

class SoftPhongShader_nm(nn.Module):
    def __init__(
        self,
        device= "cpu",
        cameras: Optional[TensorProperties] = None,
        lights: Optional[TensorProperties] = None,
        materials: Optional[Materials] = None,
        blend_params: Optional[BlendParams] = None,
    ) -> None:
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def to(self, device):
        # Manually move to device modules which are not subclasses of nn.Module
        cameras = self.cameras
        if cameras is not None:
            self.cameras = cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)
        return self

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of SoftPhongShader"
            raise ValueError(msg)
            
        try:
            mono= meshes.textures.maps_padded()[0,:,:,0].var()<1e-8#hack, so ths is actually BW, ok to set texel constant
            useNM=True and useNormalMaps
            isUV=True
        except:
            mono=False
            useNM=False#so vertex textures, for mscale hack, and never normal maps
            isUV=False
        if mono: 
            if np.random.rand()<10.5:#experiment, no nm for mono 
              useNM = False#
            texels=meshes.textures.maps_padded().view(-1,3).mean(0)#so 3channel rgb
            texels=texels.view(1,1,1,1,3)
        else:
          texels = meshes.sample_textures(fragments)
               
        materials = kwargs.get("materials", self.materials)
        if useNM:
            texels_nm0 = meshes.normal_map.sample_textures(fragments)# make sure that the normal map texture is defined!!
            texels_nm=sample_texturesN(fragments,meshes,texels_nm0[...,:3])#TODO static method,
        else:
            texels_nm=0
        lights = kwargs.get("lights", self.lights)
                
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors,nrm = phong_shading_nm(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            texels_nm=texels_nm,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )#fragment RGB colors, nrm is the optional fragment normalmap normal
        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        images = softmax_rgb_blend(colors, fragments, blend_params, znear=znear, zfar=zfar)
        
        with torch.no_grad():#avoid color leak beyond 3d figure
                mask= 1*images[:,:,:,3:4]
                mask[mask>0.1]=1
                r=torch.zeros(images.shape[0],1,1,3).uniform_(0,1).cuda()

        if fragments.pix_to_face.shape[1]<512:#N, H, W, K
            if mono and False:# or np.random.rand()<0.3:#better textures? cjance to avoud neural shader and pure UV textzre   
                if images.shape[2]<400:
                    im=images[:,:,:,:3]
                    d= fragmentD(fragments)-1 #(fragmentD(fragments)-1)/99
                    m=[]
                    with torch.no_grad():
                      for b in range(d.shape[0]):
                        m.append(d[b].max()+torch.zeros(1).cuda())
                    d/=torch.cat(m).view(-1,1,1,1)
                    #d=sobel(d,True).permute(0,2,3,1)*20
                    d=torch.cat([images,d],-1)
                    im=im+ORACLE.shaderI.dshader(d)+ORACLE.shaderI.dlshader(d)
                    return torch.cat([mask*im+(1-mask)*r,images[:,:,:,3:]],3)
                return images
        
        if useNeuralShader and ORACLE.shaderI is not None:
            nrmI=softmax_rgb_blend(nrm, fragments, blend_params, znear=znear, zfar=zfar)
            try:
                if mono:
                  sh=0.05*ORACLE.shaderI(torch.cat([images,nrmI],3).permute(0,3,1,2)).permute(0,2,3,1)
                else:
                  sh=0.05*ORACLE.shaderI.dshader(torch.cat([images,nrmI],3).permute(0,3,1,2)).permute(0,2,3,1)
            except Exception as e:
                print (e,"no neural shader")
                return images
          
            sh=sh*mask
            
            ORACLE.shadereg += (sh**2).mean()/mask.mean()
            if False and images.shape[2]<= 400:#add random background
              sh =sh+ (1-mask)*r
              #print ("rc",r)
            else:
              sh = sh + (1-mask)
            return torch.cat([mask*images[:,:,:,:3]+sh,images[:,:,:,3:]],3)
        return images


BLP = BlendParams(sigma=1e-4,gamma=1e-4)#will gamma
sigma =1e-4*5#*1e-1#*0.3#*4

raster_settings_soft = RasterizationSettings(
    image_size=224+112*0, 
    blur_radius=np.log(1. / 1e-4 - 1.)*sigma ,
    faces_per_pixel=15, perspective_correct=usePerspective and False,max_faces_per_bin=orig_v_template.shape[0]
)##from collab demo -- softer for differentbiable geometry? - -read reference!!

class MeshRendererWithDepth(nn.Module):
    def __init__(self):
        super().__init__()
        self.rasterizer = MeshRasterizer(cameras=camera, raster_settings=raster_settings_soft )
        self.shader = SoftPhongShader_nm(device=device,cameras=camera,lights=lights,blend_params=BLP)
        self.fpp = 15-5
        
    def forward(self, meshes_world,sizeExtra=64*0+96 ,**kwargs) -> torch.Tensor:
        #sigma= np.random.rand()*(1e-4-1e-6)+1e-6 #btw 1e-6 and 1e-4
        sigma=np.random.rand()*1.5
        sigma=10**(-6.+sigma)#so e-6. to e-4.
        rs=RasterizationSettings(image_size=224+sizeExtra,blur_radius=np.log(1. / 1e-4 - 1.)*sigma ,
        faces_per_pixel=self.fpp, perspective_correct=usePerspective and False,max_faces_per_bin=orig_v_template.shape[0])       
        #hmm, is the variable window too bad for memry?
        fragments = self.rasterizer(meshes_world,raster_settings=rs, **kwargs)##TODO pass raster_settings to overwrite
        images = self.shader(fragments, meshes_world, **kwargs)##is light and camera passed correctly??
        return images, fragments.zbuf
    
class MeshRendererWithDepth2(nn.Module):
    def __init__(self):
        super().__init__()
        self.rasterizer = MeshRasterizer(cameras=camera, raster_settings=raster_settings_soft )
        self.shader = SoftPhongShader(device=device,cameras=camera,lights=lights,blend_params=BLP)
        self.fpp = 28#+18
        
    def forward(self, meshes_world,sizeExtra=64*0+112 ,**kwargs) -> torch.Tensor:
        #sigma= np.random.rand()*(1e-4-1e-6)+1e-6 #btw 1e-6 and 1e-4
        sigma=np.random.rand()*1.5
        sigma=10**(-6.+sigma)#so e-6. to e-4.
        rs=RasterizationSettings(image_size=224+sizeExtra,blur_radius=np.log(1. / 1e-4 - 1.)*sigma ,
        faces_per_pixel=self.fpp, perspective_correct=usePerspective and False,max_faces_per_bin=orig_v_template.shape[0])       
        #hmm, is the variable window too bad for memry?
        fragments = self.rasterizer(meshes_world,raster_settings=rs, **kwargs)##TODO pass raster_settings to overwrite
        images = self.shader(fragments, meshes_world, **kwargs)##is light and camera passed correctly??
        return images, fragments.zbuf
    
renderer_train = MeshRendererWithDepth()
renderer_train2 = MeshRendererWithDepth2()# no shader or normal stuff, no collision detextion -- less faces!!

renderer_train_simple = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=camera, 
        raster_settings=raster_settings_soft
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=camera,
        lights=lights
    )
)#used for clip embeddings inside LSTM deformer



#inference soft too? - difference when makinf videos? what us optimal face count and sigma...

raster_settings_soft_highrescor = RasterizationSettings(
    image_size=sideX, 
    blur_radius=np.log(1. / 1e-4 - 1.)*1e-7,
    faces_per_pixel=4, perspective_correct=usePerspective,
    cull_backfaces=False,
    max_faces_per_bin=orig_v_template.shape[0]+10000#for highres
)
  
#higher res, for inference, overwrite other renderer!!
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=camera, 
        raster_settings=raster_settings_soft_highrescor
    ),
    shader=SoftPhongShader_nm(
        device=device, 
        cameras=camera,
        lights=lights, blend_params=BLP
    )
)
try:
    lights=calcLights()
    mater=calcMaterials()
except Exception as e:
    mater=Materials(shininess=100).cuda()
    lights=DirectionalLights(ambient_color=((0.2, 0.2, 0.2), ), diffuse_color=((.4, .4, .4), ),\
            specular_color=((.4, .4 ,.4), ), direction=((0, 5, 10), ), device=device)
    print (e,"light error")
  
import warnings
warnings.filterwarnings("default")   #default ("error")
tex=torch.zeros(1,TEXSIZE,TEXSIZEw,3).cuda()

#mesh3=m_check
#mesh3.textures=TexturesUV(maps=tex,faces_uvs=[faces_uvs], verts_uvs=vt)
#mesh3.normal_map=mesh3.textures
if True:#to debug cameras
    with torch.no_grad():       
      camera= getCamP(R=R2[:1], T=T2[:1])
      r=np.random.randint(R2.shape[0])
      camera= getCamP(R=R2[r:r+1], T=T2[r:r+1])
      images_ = renderer(mesh3, lights= lights,cameras=camera,materials=mater)
   #   fig=plt.figure(figsize=(10.5,10.5))
  #    plt.imshow(images_[0,:,:,:3].cpu())
 #     plt.show()

z=mesh3.verts_packed()[:,2]
print (z.shape)
#plt.plot(np.sort(z.detach().cpu().numpy()))
#plt.show()

images2_ = renderer_train(mesh3, lights= lights,cameras=camera,materials=mater)



# Rasterization settings for silhouette rendering  
sigma = 1e-5
raster_settings_silhouette = RasterizationSettings(
    image_size=256, 
    blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
    faces_per_pixel=20# Rasterization settings for silhouette rendering  
)

# Silhouette renderer 
renderer_silhouette = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=camera, 
        raster_settings=raster_settings_silhouette
    ),
    shader=SoftSilhouetteShader()
)


try:
  if True:#to debug cameras for trianing
    with torch.no_grad():
      images_,_ = renderer_train(mesh3, lights= lights,cameras=camera,materials=mater)
      #fig=plt.figure(figsize=(8,8))
      #plt.imshow(images_[0,:,:,:3].cpu())
      #plt.show()
except Exception as e:
  print (e)
  pass




def sobel(x,abs=False,sq=True):
    x=x.permute(0,3,1,2)
    x=F.interpolate(x,size=(224,224))
    assert(x.shape[1]<=4)
    border1 = x[:, :, :-1] - x[:, :, 1:]
    border1 = torch.cat([border1, x[:, :, :1] * 0], 2)  ##so square with extra 0 line
    border2 = x[:, :, :, :-1] - x[:, :, :, 1:]
    border2 = torch.cat([border2, x[:, :, :, :1] * 0], 3)
    if abs:
        border = torch.cat([border1.abs().mean(1).unsqueeze(1), border2.abs().mean(1).unsqueeze(1)], 1)##keep orientation of edge
        return border
    if sq:
        border=(border1**2 + border2**2).mean(1).unsqueeze(1)#.sqrt()
        with torch.no_grad():
          m = border.mean()
          s=border.std()
        return (border-m)/s
        #border = 1 - (-2 * (1e-5 + border).sqrt()).exp()  ##so no edge is 0, any edge quickly goes to 1
        return border
    border = border1+border2# torch.cat([border1, border2], 1)
    return border

def alignTS(a,b):
      delta_bw_color = (sobel(a)-sobel(b)).abs().mean()
      return delta_bw_color#faster


def priorT(img):#img is B H W C
        tv_h = ((img[:,:,1:] - img[:,:,:-1]).pow(2)).mean()
        tv_w = ((img[:,1:,:] - img[:,:-1,:]).pow(2)).mean()    
        return (tv_h + tv_w)


lparams = torch.cuda.FloatTensor(12).uniform_(-0.05,0.05)
lparams=nn.Parameter(lparams*5)

lparams2 = torch.cuda.FloatTensor(12).uniform_(-0.05,0.05)
lparams2=nn.Parameter(lparams2*5)

mparams = torch.cuda.FloatTensor(1,10).uniform_(-0.05,0.05)
mparams=nn.Parameter(mparams*5)

#all white -- just light enough to change colors
def calcMaterials():  
  sc=0.1
  mater=Materials(shininess=torch.sigmoid(mparams[0,9])*999)
  #mater=Materials(shininess=500)
  mater=mater.cuda()#so shininess 450 default
  return mater



def calcLights_(lparams,verb):
    if True:
        dif=  0.3+0.4*torch.sigmoid(lparams[None, 3:6])#range 0.3 to 0.7
        amb=  0.15+0.3*torch.sigmoid(lparams[None, 6:9])#range 0.2 to.5
        spec=  0.2+0.3*torch.sigmoid(lparams[None, 9:12])
        
        s=amb+dif+spec#-1e-1#TODO divide?
        amb=amb/s
        dif=dif/s
        spec=spec/s
    
    if verb:
            print ("light",amb,dif,spec)    
    
    lights=DirectionalLights(ambient_color=amb, diffuse_color=dif,
            specular_color=spec, direction=((0, 5, 5), ), device=device)
    return lights

def calcLights(verb=False):
  return calcLights_(lparams,verb)
def calcLights2(verb=False):
  return calcLights_(lparams2,verb)


#return mean and scale, size B13
def calibrateF(wc,v2):
    with torch.no_grad():
        if len(wc.shape)==2:
            wc=wc.unsqueeze(0)  
            
        if wc.shape[0] != v2.shape[0]:
            print (wc.shape,v2.shape)
            raise Exception("cam discrepancy")

        B=wc.shape[0]
        M = torch.zeros(B,1,3).cuda()
        S = torch.zeros(B,1,1).cuda()#isoscale
        for b in range(B):#TODO make batch op
            for c in range(3):
                ma = v2[b,:,c].max()
                mi = v2[b,:,c].min()
                M[b,0,c]=(ma+mi)/2
            #print ("mami sanity",wc[b,:,0].max()-wc[b,:,0].min(),wc[b,:,1].max()-wc[b,:,1].min())
            S[b,0,0] = max(wc[b,:,0].max()-wc[b,:,0].min(),wc[b,:,1].max()-wc[b,:,1].min())
    return (M,2/S)

#max interval
def MaMi0(x):
      return x.max()-x.min()
#mean
def MaMi1(x):
      return 0.5*(x.max()+x.min()).unsqueeze(0)

def sampleOrient():
  s=viewport_limit
  buf= list(range(180-s,180+s)) + list(range(360-s,360+s))
  return buf[np.random.randint(len(buf))]


from pytorch3d.transforms import Transform3d
#first version: 4 params; later add xyz rotate
def trans(v,theta=None):
  if len(v.shape) ==2:
    print ("weird unsqueeze")
    v=v.unsqueeze(0)#so BxNx3 tensor
  #print ("trans",v.shape,theta.shape,"device",v.device,theta.device)
  out=[]

  if np.random.rand()<0.15:
    print ("theta",theta[5:])

  reg=0
  for i in range(NClones):
    x=v[:,i*NVOrig:(i+1)*NVOrig]
    if False:#i==0:
      out.append(x)
    else:
      out.append(trans_(x,theta[i*5:i*5+5]))
      reg += (out[0]-out[-1]).abs().mean(-2).mean()#so close on average
  return torch.cat(out,1),reg

def trans_(v,theta):
  #out=v*theta[0].exp() + theta[1:4].view(1,1,3)
  if False:
    t1 = Transform3d().scale(theta[0].exp()).translate(theta[1],theta[2],theta[3]).cuda()
  elif False:
    t1 = Transform3d().cuda()
  else:
    tt=2
    t1 = Transform3d().translate(tt*theta[1],0*theta[2],tt*theta[3]).cuda()
    #rot_y = RotateAxisAngle(theta[0]*120,'Z').cuda()
    #t1=t1.compose(rot_y)

  rot_y = RotateAxisAngle(theta[4]*120,'Y').cuda()
  t1=t1.compose(rot_y)
  out = t1.transform_points(v)
  return out


def render_mesh(poses, textures,scaleAugm=1,tScale=0,\
               deform_value=None,rcam=None,\
               cam_rescale=False,calibrate=None,mesh_out_only=False,global_orient=None,verb=False, randomCrop=False,orient=None):  
  if rcam is None:
    rcam=sampleRC()  
  if rcam is None:
    rcam,_=sampleRC()
  num_views = len(rcam)

  if False:
    v2 = ORACLE.net(orig_v_template).unsqueeze(0) #deform    
    v2=v2.expand(num_views,-1,-1)  
    v2,reg_part=trans(v2,ORACLE.theta)
  elif False:
    v2=orig_v_template.unsqueeze(0) 
    v2,reg_part=trans(v2,ORACLE.theta)
    v2 = ORACLE.net(v2.squeeze()).unsqueeze(0) 
    v2=v2.expand(num_views,-1,-1)  
  else:
    v2=orig_v_template.unsqueeze(0) 
    
    v2 = ORACLE.net(v2.squeeze()).unsqueeze(0) 
    v2,reg_part=trans(v2,ORACLE.theta)#so sym afterwards!
    v2=v2.expand(num_views,-1,-1)  

  ORACLE.reg_part=reg_part

  if orient is None:#random yaxis rotate, azimuth
    orient = []
    for i in range(num_views):
      orient+=[sampleOrient()]
  rot_y = RotateAxisAngle(orient,'Y', device=device)
  v2= rot_y.transform_points(v2)

  faces=f2.expand(num_views,-1,-1)  

  mesh3 = Meshes(v2,faces)
  if calibrate is not None:
        me=calibrate[0]
        dx=calibrate[1]
        v2_=(v2-me)
        v2_=v2_*2/dx
        mesh3 = Meshes(v2_,faces) 
  
  if randomCrop or True:#subset of mesh for training rendered in window
    #ix=np.random.randint(low=0, high=v2.shape[1], size=(60,))
    ix = range(NVOrig)##ahaa, only first clone TODO all for camera limit! (parts are still for clone 1 only)
    ix = range(v2.shape[1])
    camera_verts=v2[:,ix]
  else:
    camera_verts=v2

  wc=rcam.transform_points(camera_verts)#mesh3.verts_packed() in ndc space already
  if cam_rescale:
    mea,sca=calibrateF(wc,camera_verts)
    v2=(v2-mea)*sca
    mesh3=Meshes(v2,faces)
      
  if mesh_out_only:
    return mesh3

  mesh3.textures=textures[0].extend(num_views)
  mesh3.normal_map=textures[1].extend(num_views)
  lights = calcLights()
  images,_ = renderer_train(mesh3, lights=lights,cameras=rcam,materials=calcMaterials())#random camera sample --- but save in canonical cameras later
  images=torch.clip(images,0,1)
  return images,mesh3

buf=[]#store losses


bRanger=True
if not bRanger:
  optimizer=torch.optim.Adam
else:
  from ranger21 import Ranger21 
  optimizer=Ranger21##hmm, more complex initialize
  #o=Ranger21()
bUNET = False##complex shading of BW
GA=1
pfr=40#frequency to print
lr=0.003*2#/GA
from time import time


text_other = '''incoherent, confusing, cropped, watermarks, grainy, pixellated, noisy'''#, degenerate limbs, weird human body


import imageio
import torchvision

def getCLIPE_(s):
  if s[-3:]=='jpg' or s[-3:]=='png':
    cpath="%s/"%(resourcePath)
    #plt.imshow(imageio.imread(cpath+s))
    #plt.show()
    img_enc = (torch.nn.functional.interpolate(torch.tensor(imageio.imread(cpath+s)).unsqueeze(0).permute(0, 3, 1, 2), (224, 224)) / 255).cuda()[:,:3]
    img_enc = clip_preprocess(img_enc)
    img_enc = perceptor.encode_image(img_enc.cuda()).detach().clone()
    print ("img prompt")
    return img_enc
  else:
    #s+= " rendered in Unreal Engine"
    tx = clip.tokenize(s)
    t = perceptor.encode_text(tx.cuda()).detach().clone()
    return t

def getCLIPE(s):
  t=[]
  for to in s.split(','):
    if len(to)==0:
        continue
    print (to)
    t.append(getCLIPE_(to))
  return t

t2=[]#empty, hack
print ("no neg words")
    
def getClipTargets(data):
    clipTargets=[]
    for t in data:
        tc0=getCLIPE(t)
        clipTargets.append(tc0)
    return clipTargets


vert1 = v10
print (vert1.shape)


bUseMscale = HRL>0 
def Scale_CLIP(m,prompts,color=0):
    total=0
    out=[]
    rcam=sampleRC()    
    v2=m.verts_padded()
    
    meshes=[]
    for t in prompts:        
        v2_=v2[:,ix_level0,:]
        #print ("mscale stuff",v2_.shape,f10.shape,len(rcam))
        mesh = Meshes(v2_,f10.expand(len(rcam),-1,-1))
        mesh.textures=TexturesVertex(verts_features=v2_*0+color) #just text_vertex
        imagesBW,_= renderer_train2(mesh, lights= calcLights(),cameras=rcam,\
                                   materials=calcMaterials(),sizeExtra=0)#.extend(num_views)

        l1bw= lossClip(imagesBW.permute(0,3,1,2)[:,:3],t,[])
        total +=l1bw
        out.append(imagesBW[0].cpu())
    return total,torch.cat(out,1)


def initParams():  
    lsc=30
    lparams.data.uniform_(-0.1*lsc,0.1*lsc)##does it work and change function?
    lparams2.data.uniform_(-0.1*lsc,0.1*lsc)
    mparams.data.uniform_(-0.1*lsc,0.1*lsc)

    bw_shade=nn.Parameter(torch.cuda.FloatTensor(1,3).uniform_(-2,2))#sigmoid afterwards anyway
    texNet = TextureL().cuda()
    mlp = PlainDeform(orig_v_template.squeeze(),initGamma=initGamma,initPV=initPV).cuda()   #orig_v_template 

    del_v=list(mlp.parameters())
    ORACLE.net = mlp
    ORACLE.last=None
    if True:
        ORACLE.shaderI=UNet(3,c_in=4+2+2,c=channels_neural_shader).cuda()
        ORACLE.shaderI.dshader = UNet(3,c_in=4+2+2,c=channels_neural_shader).cuda()
        #ORACLE.shaderI.dshader = nn.Sequential(nn.Linear(3+1+1,30),nn.ReLU(),nn.Linear(30,3)).cuda()#,nn.Sigmoid()
        #ORACLE.shaderI.dlshader =nn.Linear(3+1+1,3).cuda()
        recolor=ORACLE.shaderI#hack to use this instead
    else:
        recolor=nn.Linear(3,3)#dummy
        ORACLE.shaderI=None
    print (mlp)

    theta=torch.zeros(NClones*5).uniform_(-0.3,0.3).cuda()
    theta=nn.Parameter(theta*0.5)
  
    return bw_shade,texNet,mlp,del_v,recolor,theta 


#center camera on mesh subpart and show it fully
def getHead(mesh3,iHead,rcam=None):
      if rcam is None:
        rcam=sampleRC()

      v2=mesh3.verts_padded()        

      wc=rcam.transform_points(v2[:,iHead,:])
      mea,sca=calibrateF(wc,v2[:,iHead,:])
      v2=(v2-mea)*sca
      #mesh3=Meshes(v2,faces)
      
      #more sophisticated: only parts seen, not whole context
      mask=iHead.float().view(1,-1,1)
      if np.random.rand()<0.6:
        v2_=v2#
      else:
        v2_ = v2*mask + (v2+20)*(1-mask) #for separate meshes
      mesh4 = Meshes(v2_,mesh3.faces_padded())
      mesh4.textures=mesh3.textures
      mesh4.normal_map=mesh3.normal_map

      lights = calcLights()
      imagesHead,_ = renderer_train(mesh4, lights=lights,cameras=rcam,materials=calcMaterials(),sizeExtra=0)
      imagesHead=torch.clip(imagesHead,0,1)
      return imagesHead,rcam


def vis_iter(tx,texn,images,images_full,imbw_r,crops,buf,text_0,text_0bw):
    # plt.figure(figsize=(20,10))
    # plt.subplot(1,2,1)
    vis_nm = texn[0,:,:,:3].detach().cpu()
    nr = torch.sqrt((vis_nm**2).sum(2))
    vis_nm = vis_nm/nr.unsqueeze(2)#so norm 1, btw -1 and 1
    vis_nm = vis_nm*0.5+0.5#so 0 to 1
    # plt.imshow(vis_nm)
    # plt.title("normal maps")
    # plt.subplot(1,2,2)
    # plt.imshow(tx[0].detach().cpu().clip(0,1))
    # plt.title("textures")
    # plt.show()

    print ("time",time()-t0)
    # plt.figure(figsize=(4*8,8))
    # plt.subplot(1,4,1)
    vis11=images[...,:3].detach().cpu()
    vis11 = vis11.chunk(num_views)
    vis11=torch.cat(vis11,1).squeeze(0)
    # plt.imshow(vis11)
    # plt.title(text_0)
    # plt.subplot(1,4,2)
    # try:
    #   plt.imshow(images_full[0])
    # except Exception as e:
    #   print (e)
    # plt.title(text_0)
    # plt.subplot(1,4,3)
    vis11=imbw_r[...,:3].detach().cpu()
    vis11 = vis11.chunk(num_views)
    vis11=torch.cat(vis11,1).squeeze(0)
    # plt.imshow(vis11)
    # plt.title(text_0bw)
    # plt.subplot(1,4,4)
    if len(crops)>0:
      crops=torch.cat(crops,1)[0,:,:,:3]
    #   plt.imshow(crops.detach().cpu())

    # plt.show()
    
    a=np.array(buf)
    print (a.shape,buf[-1])
    labels=[text_0,text_0bw,"color parts","black and white parts"]
    # for i in range(4):
    #   plt.plot(a[:,i],label=labels[i])
    # plt.title('CLIP')
    # plt.legend()
    # plt.show()

    names = ['total','tex','sil']+['edge','lap','normal'] + ['deform','edgeEach',"formTexture_align","shader_regularize"]
    a=a[:,4:]
    # plt.figure(figsize=(24,3))
    # for i in range(a.shape[1]):
    #   plt.subplot(1,a.shape[1],i+1)
    #   plt.plot(a[:,i])
    #   plt.title(names[i])
    # plt.show()


def runStep(z,t0=time(),OpPa=None,clipTargets=None,dataText=None):
  bw_shade,texNet,mlp,del_v,recolor,opti,buf,imagesA = OpPa
  tc0= clipTargets[0]
  tc0b=clipTargets[1]
  text_0 = dataText[0]
  text_0bw = dataText[1]
    
  torch.cuda.empty_cache()
  if z%GA==0:
    opti.zero_grad()  #what is this? HACK MAy 10 inspect 
    ORACLE.shadereg=0*(tc0[0]).sum()

  if True:
    tx,texn=texNet()#.permute(0,2,3,1)
    #if z<3:
      #plt.imshow(tx[0].detach().cpu())
      #plt.show()
    textures =TexturesUV(maps=tx,faces_uvs=[faces_uvs], verts_uvs=vt)
    texturesn =TexturesUV(maps=texn,faces_uvs=[faces_uvs], verts_uvs=vt)

  rcam=sampleRC()
  images,mesh3=render_mesh(None,(textures,texturesn),\
                                     deform_value=0,rcam=rcam,cam_rescale=True,randomCrop=True)##hack: was True -- if False will also optimize pose, allowing to cheat but get rid of dynamics!!
  #print ("runstep done")
  deform_value = mlp.last
  l1= lossClip(procClip(images),tc0,t2)
  if z <20:
      print ("l1 loss",l1.item(),len(tc0),tc0[0].shape,text_0)
  reg_siam0=l1*0
  
  texturesBW=TexturesUV(maps=tx.detach()*0+torch.sigmoid(bw_shade),faces_uvs=[faces_uvs], verts_uvs=vt)
  mesh3.textures=texturesBW.extend(num_views)
  imagesBW,depthBW = renderer_train(mesh3, lights= calcLights2(),cameras=rcam,materials=calcMaterials(),sizeExtra=0)#random camera sample --- but save in canononical cameras later
  mesh3.textures=textures
  depthBW=imagesBW

  l1bw= lossClip(procClip(imagesBW),tc0b,[])
  imbw_r=imagesBW#.permute(0,3,1,2)[:,:3]
  delta_bw_color = alignTS(imbw_r,images)
    
  if bUseMscale:# and False:
      l1bw_scale,out_scale=Scale_CLIP(mesh3,[tc0b],torch.sigmoid(bw_shade))
      l1bw=l1bw+l1bw_scale
  

  def getSubLoss(iHead,tc1,tc1b,sizeExtra=0):
      mesh3.textures=textures.extend(num_views)
      imagesHead,rcam=getHead(mesh3,iHead)
      l1head=lossClip(procClip(imagesHead),tc1,t2)
      mesh3.textures=texturesBW.extend(num_views)
      imagesHead_bw,_=getHead(mesh3,iHead,rcam=rcam)
      delta_bw_color3 = alignTS(imagesHead,imagesHead_bw)
      silh_hbw=imagesHead_bw[:,:,:,3:4]                                        
      l1head_bw = lossClip(procClip(imagesHead_bw),tc1b,[])
      with torch.no_grad():
          if imagesHead.shape[2]!=224:
            imagesHead =F.interpolate(imagesHead.permute(0,3,1,2),size=(224,224)).permute(0,2,3,1)
          crop1=torch.cat([imagesHead[:,:,:],imagesHead_bw[:,:,:]],2)      
      return l1head,l1head_bw,crop1

  loss_aux=l1*0
  loss_auxbw=l1*0
  crops=[]
  for i in range(1,len(clipTargets)//2):
    iHead=parts[i-1]
    tc1=clipTargets[i*2]
    tc1b=clipTargets[i*2+1]
    lpart,lpartbw,crop1=getSubLoss(iHead,tc1,tc1b,sizeExtra=0)
    loss_aux+=lpart
    loss_auxbw+=lpartbw
    crops.append(crop1)

  mesh3.textures=textures
  silh=images[:,:,:,3]##soft silhouette is differentiable.. 
 
  l_e = (mesh_edge_loss(mesh3))# -l_e_orig).abs()
  l_l = l_e*0#(mesh_laplacian_smoothing(mesh3))#-l_l_orig).abs()
  l_n = (mesh_normal_consistency(mesh3))#-l_n_orig).abs
  l_orig = (deform_value**2).mean()
  l3 =priorT(tx)

  l5= ((silh.mean(2).mean(1)-fSilhTarget)**2).mean()
   
  gen_edge_len= mesh_edge_len(mesh3)[:edge_len.shape[0]]  
  loss_edge_len = ((wel*(edge_len-gen_edge_len ))**2).mean()
  loss = l1bw+ l1+1e-3*l3+fSilhStrength *l5+\
  4e0*(1e-1*l_e+50*0.1*l_n+0*l_l)+fPenaltyDeformation*l_orig+fPenaltyDeformation_e*loss_edge_len 

  loss+=loss_aux+loss_auxbw
  loss += 1e-1*(delta_bw_color)
  loss += fShader_reg*ORACLE.shadereg
  
  loss +=fPartRegularize*ORACLE.reg_part
  print ("reg part",ORACLE.reg_part)

  loss.backward()
  buf.append([l1.item(),l1bw.item(),loss_aux.item(),loss_auxbw.item(),loss.item(),l3.item(),silh.mean().item(),l_e.item(),l_l.item(),l_n.item(),l_orig.item(),loss_edge_len.item(),\
              delta_bw_color.item(),ORACLE.shadereg.item()])
  
  if z<=5 or z%10==0:
    print (z,"total",loss.item(),"texsm",l3.item(),"silh",silh.mean().item(),\
       "mesh",l_e.item(),l_l.item(),l_n.item(),"off-mesh",l_orig.item())
    
    print ("dbwcoolor ",delta_bw_color.item())
    print ("clip losses",l1.item(),l1bw.item())
    print ("shader reg",ORACLE.shadereg.item())
    print ("")
    
  best =loss.item()
  if z%GA ==GA-1:
    opti.step()
    if z%1==0:#save training frame for output animation
      with torch.no_grad():       
          target_cameras =getCamP(R=R2[:1], T=T2[:1])#choose 1 camera, but larger res
          mesh3=render_mesh(None,(textures,texturesn),cam_rescale=True,mesh_out_only=True,orient=0,rcam=target_cameras)
          v =mesh3.verts_packed()
          mesh3.textures=textures#.extend(1)
          mesh3.normal_map=texturesn#.extend(1)
          ci = (z//GA)%R2.shape[0]#iterate cameras, rotate during saving
          target_cameras =getCamP(R=R2[ci:ci+1], T=T2[ci:ci+1])#choose 1 camera, but larger res
          lights = calcLights()
          images_full = renderer(mesh3, lights= lights,cameras=target_cameras,materials=calcMaterials()).clamp(0,1).cpu()[:,:,:,:3]
          imagesA.append(images_full.half())#
    
  if z%pfr==0 or z <5:
    vis_iter(tx,texn,images,images_full,imbw_r,crops,buf,text_0,text_0bw)
  return images,imagesBW



def getCalibrate(textures,texturesn):
    with torch.no_grad():##just any camera for roughly centering mesh
      mesh3=render_mesh(None,( textures,texturesn),mesh_out_only=True)
      ci=0
      rcam =getCamP(R=R2[ci:ci+1], T=T2[ci:ci+1])
      camera_verts=mesh3.verts_packed()#[:NVOrig]#v2
      wc=rcam.transform_points(camera_verts).unsqueeze(0)#if 1 camera otherwise jusdt 2d tensor
      v2=mesh3.verts_padded()[:,:NVOrig]
      print (v2.shape,v2.max(),v2.min(),"wc",wc.shape,wc.max(),wc.min())
      dx= MaMi0(wc[:,:,0])
      dy= MaMi0(wc[:,:,1])
      me0=MaMi1(v2[:,:,0])
      me1=MaMi1(v2[:,:,1])
      me2=MaMi1(v2[:,:,2])
      ##TODO average ovrr multiple frames
      me=torch.cat([me0,me1+0,me2]).view(1,1,3)      
      calibrate=(me,max(dx,dy)+0.15)
      return calibrate
    
def save_frames(textures,texturesn,calibrate =None,Ta=None,Sa=None,Ae=None,rev=False):
    os.system('rm frames2/*')
    if calibrate is None:
        calibrate= getCalibrate(textures,texturesn)
        
    corig = calibrate
    calibrate=None    
    imagesA=[]
    print ("starting buf",len(imagesA))
    ci=0
    T=360#time length of output video
    if True:  
      for t in range(T):#
          if Ta is not None:
              m=corig[0]*1
              s = corig[1]*1  
              alfa=t/float(T-1)#from 0 to 1
              if rev:
                    alfa=1-alfa
              alfa = alfa**Ae#so more aggressive
              m[...,1]+= alfa*Ta #Y axis towards gead
              calibrate =(m,s/(1+Sa*alfa))
          with torch.no_grad():
            ci = (ci+1)%R2.shape[0]#iterate cameras
            target_cameras=getCamP(R=R2[ci:ci+1], T=T2[ci:ci+1])
            
            mesh3=render_mesh(None,(textures,texturesn),calibrate=calibrate,mesh_out_only=True,orient=t-60,rcam=target_cameras,cam_rescale=True)
            mesh3.normal_map=texturesn#last call from training
            
            if True:
              mesh3.textures=textures                 
              images_ = renderer(mesh3, lights=calcLights(),materials=calcMaterials(),cameras=target_cameras)
              if False:
                 images_=bwWindow(mesh3,images_,target_cameras,calcLights2(),z=ztarget)     
            else:
              mesh3.textures=texturesBW
              imagesBW = renderer(mesh3, lights=calcLights2(),cameras=target_cameras,materials=calcMaterials())
              images_=imagesBW

            imagesA.append(images_.clamp(0,1).cpu().half())

            # if t%60==0:
            #     plt.imshow(images_[0,:,:,:3].detach().cpu())
            #     plt.title("motion step %s"%(t))
            #     plt.show()
    return imagesA,mesh3

def director(final_name,textures,texturesn,Ta = 0.5,Sa=1.35,Ae=0.4,rev=True,imagesA=None):
    if imagesA is not None:
        os.system('rm frames2/*')
        mesh3 = None
    else:
        imagesA,mesh3=save_frames(textures,texturesn,Ta = Ta,Sa=Sa,Ae=Ae,rev=rev)
        for i in range(len(imagesA)):
          im=imagesA[i]
          if im.shape[3]==4:
            imagesA[i] = imagesA[i][:,:,:,:3]
    saveImageSet(mesh3,imagesA,"t")
    
    mname = "movier4flat.mov"
    delCmd = "rm %s"%(mname)
    os.system(delCmd)
    
    mCmd = 'ffmpeg -framerate 24 -r 24 -i "%s'%("frames2") +  '/%d.jpg"'
    mCmd += " -c:v libx264 -crf 24 -pix_fmt yuv420p %s"%(mname)
    os.system(mCmd)    
    shutil.copy(mname,final_name)
    return mesh3



#@title Create!
from IPython.display import clear_output
for nText in range(1):
    clear_output(wait=True)
    dataText,descriptor=getTexts()##
    clipTargets=getClipTargets(dataText)
    bw_shade,texNet,mlp,del_v,recolor,theta = initParams()
    ORACLE.theta=theta
    lr=0.003
    opti = optimizer([{'params': [lparams,lparams2,mparams,bw_shade], 'lr':lr},\
                    {'params': del_v+ list(recolor.parameters()), 'lr':4e-3},\
                    {'params': theta, 'lr':1e-1},\
                    {'params': list(texNet.parameters()), 'lr':4e-1}], lr=lr,num_epochs =1,num_batches_per_epoch=GA*optisteps)
    OpPa = (bw_shade,texNet,mlp,del_v,recolor,opti,[],[])  
    
    t0 = time()
    for z in range(optisteps*GA):
        images,imagesBW =runStep(z,t0,OpPa,clipTargets,dataText)#output just for vis
        
    ##when moved to function -- some stuff needs to be prepared again
    ci=0
    target_cameras =getCamP(R=R2[ci:ci+1], T=T2[ci:ci+1])
    tx,texn=texNet()#.permute(0,2,3,1)
    textures =TexturesUV(maps=tx,faces_uvs=[faces_uvs], verts_uvs=vt)
    texturesn =TexturesUV(maps=texn,faces_uvs=[faces_uvs], verts_uvs=vt)
    texturesBW=TexturesUV(maps=tx.detach()*0+torch.sigmoid(bw_shade),faces_uvs=[faces_uvs], verts_uvs=vt)
    ORACLE.last=mlp.last
    final_name="%s/CanonicS_%s_sym%s_NE%d.mov"%(vid_dir,descriptor.replace('.','_'),mlp.bSYM,NE,)

    if(create_video):
      #video 1: rotate in full size visible
      mesh3=director(final_name,textures,texturesn,Ta = None,Sa=None,Ae=0.4,rev=True)
      #video 2: rotate and zoom from top
      #mesh3=director(final_name.replace("CanonicS","movZoomS"),textures,texturesn,Ta = 0.55,Sa=1.35,Ae=0.75,rev=True)
      #video 3: show evolution of trained mesh
      _=director(final_name.replace("CanonicS","iteration"),textures,texturesn,imagesA=OpPa[-1])


    if(bsave_obj):
      save_obj('%s.obj'%(descriptor,),verts=mesh3.verts_packed(),faces=mesh3.faces_packed(),\
                 verts_uvs = vt.squeeze(),faces_uvs=faces_uvs.squeeze(),texture_map=textures.maps_padded().squeeze())
      shutil.move(descriptor+'.obj', f'{meshDirPath}/{descriptor}.obj')

    if(save_texture):
      img=np.uint8(tx[0].cpu().detach().clip(0,1)*255)#tx and not tex due to sigmoid and other mods
      imageio.imwrite(descriptor+'texture.png',img)
      shutil.move(descriptor+'texture.png', f'{meshDirPath}/{descriptor}_texture.png')


# if(show_3d_interactive_runall):
#   #mesh3.textures=texturesBW
#   fig = plot_scene({
#       "text_to_add": {
#           "mean optimized": mesh3
#       }
#   })
#   fig.show()


mesh3=director(final_name,textures,texturesn,Ta = None,Sa=None,Ae=0.4,rev=True)
#iHead.shape
iHead[:10].float()
a=torch.zeros(1000,10)
print (a.mean(0,keepdim=True).shape)