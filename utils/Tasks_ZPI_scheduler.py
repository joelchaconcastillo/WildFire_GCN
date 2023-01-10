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
from pathlib import Path
import os
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import networkx as nx
import zigzag.zigzagtools as zzt
import zigzag.ZZgraph as zzgraph
from scipy.spatial.distance import squareform
import scipy.sparse as sp
import dionysus as d
import time
from ripser import ripser
import scipy.sparse
##import argparse
##
###from persim import plot_diagrams, PersImage
##path = os.getcwd()
##args = argparse.ArgumentParser(description='arguments')
##args.add_argument('--source', default='.', type=str)
##args.add_argument('--dest', default='.', type=str)
##args.add_argument('--scaleParameter', default='.', type=float)
##
##args = args.parse_args()
##print(args.source)
##print(args.dest)
##print(args.scaleParameter)
##exit(0)
#####TDA parameters
maxDimHoles = 2
window = 10
alpha = 1
scaleParameter =  0.1
sizeBorder = 6
NVertices = (2*sizeBorder+1)**2

class loadData(Dataset):
   def __init__(self, dataset_root: str = None, access_mode: str = 'spatiotemporal', dynamic_features: list = None, static_features : list = None, nan_fill: float= -1., clc: str=None):

      assert access_mode in ['spatial', 'temporal', 'spatiotemporal']
   
      if not dataset_root:
         raise ValueError('dataset_root variable must be set. Check README')
   
      dataset_root = Path(dataset_root)
      min_max_file = dataset_root / 'minmax_clc.json'
      variable_file = dataset_root / 'variable_dict.json'
   
      ##Opening max-min information of features..
      with open(min_max_file) as f:
         self.min_max_dict = json.load(f)
   
      with open(variable_file) as f:
         self.variable_dict = json.load(f)
   
      dataset_path = dataset_root / 'npy' / access_mode
      self.dynamic_features = dynamic_features
      self.static_features = static_features
      self.clc = clc
      self.nan_fill = nan_fill
      self.access_mode = access_mode
      self.path_list = list((dataset_path / 'positives').glob('*dynamic.npy'))+list((dataset_path / 'negatives_clc').glob('*dynamic.npy'))
   
      self.dynamic_idxfeat = [(i, feat) for i, feat in enumerate(self.variable_dict['dynamic']) if feat in self.dynamic_features]
      self.static_idxfeat = [(i, feat) for i, feat in enumerate(self.variable_dict['static']) if feat in self.static_features]
      self.dynamic_idx = [x for (x, _) in self.dynamic_idxfeat]
      self.static_idx = [x for (x, _) in self.static_idxfeat]
#      print(self.path_list)
      print("Dataset length", len(self.path_list))
      self.mm_dict = self._min_max_vec()

   def _min_max_vec(self):
      mm_dict = {'min': {}, 'max': {}}
      for agg in ['min', 'max']:
         if self.access_mode == 'spatial':
            mm_dict[agg]['dynamic'] = np.ones((len(self.dynamic_features), 1, 1))
            mm_dict[agg]['static'] = np.ones((len(self.static_features), 1, 1))
            for i, (_, feat) in enumerate(self.dynamic_idxfeat):
               mm_dict[agg]['dynamic'][i, :, :] = self.min_max_dict[agg][self.access_mode][feat]
            for i, (_, feat) in enumerate(self.static_idxfeat):
               mm_dict[agg]['static'][i, :, :] = self.min_max_dict[agg][self.access_mode][feat]

         if self.access_mode == 'temporal':
            mm_dict[agg]['dynamic'] = np.ones((1, len(self.dynamic_features)))
            mm_dict[agg]['static'] = np.ones((len(self.static_features)))
            for i, (_, feat) in enumerate(self.dynamic_idxfeat):
               mm_dict[agg]['dynamic'][:, i] = self.min_max_dict[agg][self.access_mode][feat]
            for i, (_, feat) in enumerate(self.static_idxfeat):
               mm_dict[agg]['static'][i] = self.min_max_dict[agg][self.access_mode][feat]
         
         if self.access_mode == 'spatiotemporal':
            mm_dict[agg]['dynamic'] = np.ones((1, len(self.dynamic_features), 1, 1))
            mm_dict[agg]['static'] = np.ones((len(self.static_features), 1, 1))
            for i, (_, feat) in enumerate(self.dynamic_idxfeat):
               mm_dict[agg]['dynamic'][:, i, :, :] = self.min_max_dict[agg][self.access_mode][feat]
            for i, (_, feat) in enumerate(self.static_idxfeat):
               mm_dict[agg]['static'][i, :, :] = self.min_max_dict[agg][self.access_mode][feat]
      return mm_dict

   def __getitem__(self, idx):
      path = self.path_list[idx]
      path = Path(path)
      dirPath = Path(*path.parts[:-1])
      name = path.stem.split('_')[:-1]
      prefix = dirPath / '_'.join(name)
      return str(prefix)

   def __len__(self):
      return len(self.path_list)

   # Zigzag persistence image
   def zigzag_persistence_images(self, dgms, resolution = [50,50], return_raw = False, normalization = True, bandwidth = 1., power = 1., dimensional = 0):
       if len(dgms) < dimensional: #validation
           return np.zeros(resolution)
       print("dimension....", dimensional)
       PXs, PYs = np.vstack([dgm[:, 0:1] for dgm in dgms]), np.vstack([dgm[:, 1:2] for dgm in dgms])
       print("number selectors..", PXs.shape)
       print(len(dgms))
       xm, xM, ym, yM = PXs.min(), PXs.max(), PYs.min(), PYs.max()
       x = np.linspace(xm, xM, resolution[0])
       y = np.linspace(ym, yM, resolution[1])
       X, Y = np.meshgrid(x, y)
       Zfinal = np.zeros(X.shape)
       X, Y = X[:, :, np.newaxis], Y[:, :, np.newaxis]
       # Compute zigzag persistence image
       P0, P1 = np.reshape(dgms[int(dimensional)][:, 0], [1, 1, -1]), np.reshape(dgms[int(dimensional)][:, 1], [1, 1, -1])
       weight = np.abs(P1 - P0)
       distpts = np.sqrt((X - P0) ** 2 + (Y - P1) ** 2)
   
       if return_raw:
           lw = [weight[0, 0, pt] for pt in range(weight.shape[2])]
           lsum = [distpts[:, :, pt] for pt in range(distpts.shape[2])]
       else:
           weight = weight ** power
           Zfinal = (np.multiply(weight, np.exp(-distpts ** 2 / bandwidth))).sum(axis=2)
   
       output = [lw, lsum] if return_raw else Zfinal
       if normalization:
           if np.max(output)-np.min(output) == 0:
               return output
           norm_output = (output - np.min(output))/(np.max(output) - np.min(output))
       else:
           norm_output = output
#       print(norm_output) 

#       fig, (ax1, ax2) = plt.subplots(1,2)
#       fig.suptitle('TDA rules! '+str(dimensional)+'-dimensional ZPD')
#       ax1.set_xlim(0, window-1)
#       ax1.set_ylim(0, window-1)
#       ax1.set_xlabel('birth')
#       ax1.set_ylabel('death')
#       ax1.plot(PXs, PYs, 'ro')
#   
#       X, Y = np.meshgrid(x, y)
#       ax2.set_xlim(xm, xM)
#       ax2.set_ylim(ym, yM)
#       ax2.contourf(X, Y, norm_output)
#       plt.savefig('my_plot'+str(dimensional)+'.pdf')
#       plt.show()
#
#       from PIL import Image
#       img = Image.fromarray(norm_output, 'L')
#       img.save('sample.png')

       return norm_output
#####################################################################
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

#data = np.random.rand(window, NVertices, 25)

###We can ge more plots from here...
###zzgraph.plotting(data, NVertices, alpha, scaleParameter, maxDimHoles, window)
### for each sample....

data = loadData(dataset_root = dataset_root,access_mode = access_mode, dynamic_features = dynamic_features, static_features = static_features, clc=clc)

#print("Number of samples", len(data))
#print("Dynamic shape..", data[0][0].shape)
#print("Static shape..", data[0][1].shape)
#print("clc shape..", data[0][2].shape)
cont = 0
pwd='/home/joel.chacon/tmp/ZIGZAG_from_files/'

for (prefix_path) in data:
   sourcePath = prefix_path
   prefix_path = '/'.join(prefix_path.split('/')[4:])
   prefix_path = 'data/'+prefix_path
   print("python3 "+pwd+"/utils/singleFile_ZPI_PD.py --source="+sourcePath + " --dest="+pwd+"/"+prefix_path+" --scaleParameter="+str(scaleParameter)+" --maxDimHoles="+str(maxDimHoles)+" --sizeBorder="+str(sizeBorder))
