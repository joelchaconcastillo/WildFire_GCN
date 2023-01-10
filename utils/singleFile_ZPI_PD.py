import sys
import os
sys.path.append('/home/joel.chacon/.local/lib/python3.8/site-packages')
#sys.path.append('/home/joel.chacon/.local/lib/python3.8/site-packages/matplotlib/')
sys.path.append('/home/joel.chacon/.local/bin')
#os.environ['MATPLOTLIBRC'] = '/home/joel.chacon/.local/lib/python3.8/site-packages/matplotlib/matplotlibrc'
#print(sys.path)
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import warnings
import json
import scipy.sparse
import pandas as pd
import zigzag.zigzagtools as zzt
from scipy.spatial.distance import squareform
import dionysus as d
import time
path = os.getcwd()
import argparse
from zigzagTDA import zigzagTDA

#from persim import plot_diagrams, PersImage
path = os.getcwd()
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--source', default='.', type=str)
args.add_argument('--dest', default='.', type=str)
args.add_argument('--scaleParameter', default='.', type=float)
args.add_argument('--maxDimHoles', default='.', type=int)
args.add_argument('--sizeBorder', default='.', type=int)

args = args.parse_args()


##########GLOBAL VARIABLES##############################################
   # Zigzag persistence image
#####################################################################
#####TDA parameters
maxDimHoles = args.scaleParameter
window = 10
scaleParameter =  args.scaleParameter
sizeBorder = args.sizeBorder
NVertices = (2*sizeBorder+1)**2

dynamic_features = [
    '1 km 16 days NDVI',
#    '1 km 16 days EVI',
#    'ET_500m',
    'LST_Day_1km',
    'LST_Night_1km',
#    'Fpar_500m',
#    'Lai_500m',
#    'era5_max_u10',
#    'era5_max_v10',
    'era5_max_d2m',
    'era5_max_t2m',
    'era5_max_sp',
    'era5_max_tp',
#    'era5_min_u10',
#    'era5_min_v10',
#    'era5_min_d2m',
#    'era5_min_t2m',
#    'era5_min_sp',
#    'era5_min_tp',
#    'era5_avg_u10',
#    'era5_avg_v10',
#    'era5_avg_d2m',
#    'era5_avg_t2m',
#    'era5_avg_sp',
#    'era5_avg_tp',
#    'smian',
    'sminx',
#    'fwi',
#    'era5_max_wind_u10',
#    'era5_max_wind_v10',
    'era5_max_wind_speed',
#    'era5_max_wind_direction',
#    'era5_max_rh',
    'era5_min_rh',
#    'era5_avg_rh',
]


static_features = [
 'dem_mean',
# 'aspect_mean',
 'slope_mean',
# 'roughness_mean',
 'roads_distance',
 'waterway_distance',
 'population_density',
]



clc = 'vec'
access_mode = 'spatiotemporal'
nan_fill = -1.0 
dataset_root = '/home/joel.chacon/tmp/datasets_grl'

############################################################
def _min_max_scaling(in_vec, max_vec, min_vec):
   return (in_vec - min_vec) / (max_vec - min_vec)  ###Warning zero divisions!!!


if not dataset_root:
   raise ValueError('dataset_root variable must be set. Check README')
   
min_max_file = dataset_root +'/minmax_clc.json'
variable_file = dataset_root +'/variable_dict.json'
   
##Opening max-min information of features..
with open(min_max_file) as f:
   min_max_dict = json.load(f)
   
with open(variable_file) as f:
   variable_dict = json.load(f)
   
###load features...
dynamic_idxfeat = [(i, feat) for i, feat in enumerate(variable_dict['dynamic']) if feat in dynamic_features]
static_idxfeat = [(i, feat) for i, feat in enumerate(variable_dict['static']) if feat in static_features]
dynamic_idx = [x for (x, _) in dynamic_idxfeat]
static_idx = [x for (x, _) in static_idxfeat]
mm_dict = {'min': {}, 'max': {}}
for agg in ['min', 'max']:
   mm_dict[agg]['dynamic'] = np.ones((1, len(dynamic_features), 1, 1))
   mm_dict[agg]['static'] = np.ones((len(static_features), 1, 1))
   for i, (_, feat) in enumerate(dynamic_idxfeat):
      mm_dict[agg]['dynamic'][:, i, :, :] = min_max_dict[agg][access_mode][feat]
   for i, (_, feat) in enumerate(static_idxfeat):
      mm_dict[agg]['static'][i, :, :] = min_max_dict[agg][access_mode][feat]

dynamic = np.load(args.source+'_dynamic.npy')  #TODO:  
static = np.load(args.source+'_static.npy')  #TODO:  
#static = np.load(str(path).replace('dynamic', 'static'))
dynamic = dynamic[:, dynamic_idx, ...]
static = static[static_idx]


dynamic = _min_max_scaling(dynamic, mm_dict['max']['dynamic'], mm_dict['min']['dynamic'])
static = _min_max_scaling(static, mm_dict['max']['static'], mm_dict['min']['static'])
feat_mean = np.nanmean(dynamic, axis=(2, 3))
feat_mean = feat_mean[..., np.newaxis, np.newaxis]
feat_mean = np.repeat(feat_mean, dynamic.shape[2], axis=2)
feat_mean = np.repeat(feat_mean, dynamic.shape[3], axis=3)
dynamic = np.where(np.isnan(dynamic), feat_mean, dynamic)

##TODO this is better...
#feat_mean = np.nanmean(dynamic, axis=0)
#feat_mean = feat_mean[np.newaxis, ...]
#feat_mean = np.repeat(feat_mean, dynamic.shape[0], axis=0)
#dynamic = np.where(np.isnan(dynamic), feat_mean, dynamic)

if nan_fill:
   dynamic = np.nan_to_num(dynamic, nan=nan_fill)
   static = np.nan_to_num(static, nan=nan_fill)
if clc == 'mode':
   clc = np.load(str(path).replace('dynamic', 'clc_mode'))
elif clc == 'vec':
   clc = np.load(args.source+'_clc_vec.npy') #p.load(str(path).replace('dynamic', 'clc_vec'))
   clc = np.nan_to_num(clc, nan=0)
else:
   clc = 0

(sizeWindow, _ , patchWidth, patchHeight) = dynamic.shape
#numberFeatures = len(dynamic_features) #+len(static_features)+len(clc)
numberFeatures = len(dynamic_features)+len(static_features)+len(clc)
sample = np.zeros((sizeWindow, NVertices, numberFeatures))
for t in range(sizeWindow):
   X = np.concatenate((dynamic[t], static, clc), axis=0) ##F, W, H
   #X = dynamic[t] #np.concatenate((dynamic[t], static, clc), axis=0) ##F, W, H
      #X = np.concatenate((dynamic[t], static), axis=0) ##F, W, H
   X = X[:,12-sizeBorder:13+sizeBorder,12-sizeBorder:13+sizeBorder]
   X = X.reshape(numberFeatures, -1) # F, N
   sample[t] = X.transpose(1,0) #N, F
#print(sample)
#print(args.dest)
ZZ = zigzagTDA(NVertices, args.scaleParameter, args.maxDimHoles, sizeWindow)
zigzag_PD = ZZ.zigzag_persistence_diagrams(x = sample, prefix_path=args.dest)
zigzag_PI_H0 = ZZ.zigzag_persistence_images(zigzag_PD, dimensional = 0)
zigzag_PI_H1 = ZZ.zigzag_persistence_images(zigzag_PD, dimensional = 1)
ZPI = [zigzag_PI_H0, zigzag_PI_H1]
#print(len(ZPI))
#np.savez(args.dest+"_zpi_"+"scaleParameter_"+str(args.scaleParameter)+"_maxDimHoles_"+str(args.maxDimHoles)+"_sizeBorder_"+str(args.sizeBorder), zpi=ZPI)

