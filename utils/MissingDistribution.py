import sys
import os
#sys.path.append('/home/joel.chacon/.local/lib/python3.8/site-packages')
#sys.path.append('/home/joel.chacon/.local/lib/python3.8/site-packages/matplotlib/')
#sys.path.append('/home/joel.chacon/.local/bin')
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
import matplotlib.pyplot as plt
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
maxDimHoles = 1
window = 10
alpha = 1
scaleParameter =  1.
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
   def combine_dynamic_static_inputs(self, dynamic, static, clc):
        '''
           dynamic: T, F, W, H
        '''
        timesteps, F, W, H = dynamic.shape
#        static = static.unsqueeze(dim=0)
        static = np.expand_dims(static, axis=0)
        #repeat_list = [1 for _ in range(static.dim())]
        repeat_list = [1 for _ in range(static.ndim)]
        repeat_list[0] = timesteps
        static = np.tile(static, repeat_list)
#        static = static.repeat(repeat_list)
        input_list = [dynamic, static]
        if clc is not None:
           #clc = clc.unsqueeze(dim=0).repeat(repeat_list)
           clc = np.expand_dims(clc, axis=0)
           clc = np.tile(clc, repeat_list)
           input_list.append(clc)
#        inputs = torch.cat(input_list, dim=1).float()
        inputs = np.concatenate(input_list, axis=1)
        return inputs.reshape(*inputs.shape[:-2], -1)  # flatten patche

   def __getitem__(self, idx):
        path = self.path_list[idx]
        dynamic = np.load(path)
        static = np.load(str(path).replace('dynamic', 'static'))
        if self.access_mode == 'spatial':
            dynamic = dynamic[self.dynamic_idx]
            static = static[self.static_idx]
        elif self.access_mode == 'temporal':
            dynamic = dynamic[:, self.dynamic_idx, ...]
            static = static[self.static_idx]
        else:
            dynamic = dynamic[:, self.dynamic_idx, ...]
            static = static[self.static_idx]

        def _min_max_scaling(in_vec, max_vec, min_vec):
            return (in_vec - min_vec) / (max_vec - min_vec)

        dynamic = _min_max_scaling(dynamic, self.mm_dict['max']['dynamic'], self.mm_dict['min']['dynamic'])
        static = _min_max_scaling(static, self.mm_dict['max']['static'], self.mm_dict['min']['static'])

        if self.access_mode == 'temporal':
            feat_mean = np.nanmean(dynamic, axis=0)
            # Find indices that you need to replace
            inds = np.where(np.isnan(dynamic))
            # Place column means in the indices. Align the arrays using take
            dynamic[inds] = np.take(feat_mean, inds[1])

#        elif self.access_mode == 'spatiotemporal':
#            with warnings.catch_warnings():
#                warnings.simplefilter("ignore", category=RuntimeWarning)
#                feat_mean = np.nanmean(dynamic, axis=(2, 3))
#                feat_mean = feat_mean[..., np.newaxis, np.newaxis]
#                feat_mean = np.repeat(feat_mean, dynamic.shape[2], axis=2)
#                feat_mean = np.repeat(feat_mean, dynamic.shape[3], axis=3)
#                dynamic = np.where(np.isnan(dynamic), feat_mean, dynamic)
#        if self.nan_fill:
#            dynamic = np.nan_to_num(dynamic, nan=self.nan_fill)
#            static = np.nan_to_num(static, nan=self.nan_fill)

        if self.clc == 'mode':
            clc = np.load(str(path).replace('dynamic', 'clc_mode'))
        elif self.clc == 'vec':
            clc = np.load(str(path).replace('dynamic', 'clc_vec'))
            clc = np.nan_to_num(clc, nan=0)
        else:
            clc = 0
        _, W, H = clc.shape
        data = self.combine_dynamic_static_inputs(dynamic, static, clc)
        return data

   def __len__(self):
      return len(self.path_list)

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
#dataset_root = '/home/joel.chacon/tmp/datasets_grl'
dataset_root = '/home/joel.chacon/tmp/selectingSampling/data/datasets_grl1/datasets_grl'
#data = np.random.rand(window, NVertices, 25)

###We can ge more plots from here...
###zzgraph.plotting(data, NVertices, alpha, scaleParameter, maxDimHoles, window)
### for each sample....

data = loadData(dataset_root = dataset_root,access_mode = access_mode, dynamic_features = dynamic_features, static_features = static_features, clc=clc)

#print("Number of samples", len(data))
#print("Dynamic shape..", data[0][0].shape)
#print("Static shape..", data[0][1].shape)
#print("clc shape..", data[0][2].shape)
T=10
H=25
W=25
F=25
N=W*H
naTimes= np.zeros((len(data), T))
count=0
for (X) in data:
    print('sample: ',count)
    for t in range(T):
        for f in range(F):
           for n in range(N):
                   if np.isnan(X[t, f, n])==True:
                      naTimes[count, t] +=1
    count+=1
    if (count % 100) == 0:
       plt.boxplot(naTimes)
       plt.savefig("NANS.pdf", format="pdf", bbox_inches="tight")
plt.savefig("NANS.pdf", format="pdf", bbox_inches="tight")
plt.show()

