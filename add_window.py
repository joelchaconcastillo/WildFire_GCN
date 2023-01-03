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
#import networkx as nx
import zigzag.zigzagtools as zzt
import zigzag.ZZgraph as zzgraph
from scipy.spatial.distance import squareform
import scipy.sparse as sp
import dionysus as d
import matplotlib.pyplot as plt
import time
from ripser import ripser
from persim import plot_diagrams, PersImage
path = os.getcwd()

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
         return (in_vec - min_vec) / (max_vec - min_vec)  ###Warning zero divisions!!!

      dynamic = _min_max_scaling(dynamic, self.mm_dict['max']['dynamic'], self.mm_dict['min']['dynamic'])
      static = _min_max_scaling(static, self.mm_dict['max']['static'], self.mm_dict['min']['static'])

      if self.access_mode == 'temporal':
         feat_mean = np.nanmean(dynamic, axis=0)
         # Find indices that you need to replace
         inds = np.where(np.isnan(dynamic))
         # Place column means in the indices. Align the arrays using take
         dynamic[inds] = np.take(feat_mean, inds[1])
      elif self.access_mode == 'spatiotemporal':
         with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            feat_mean = np.nanmean(dynamic, axis=(2, 3))
            feat_mean = feat_mean[..., np.newaxis, np.newaxis]
            feat_mean = np.repeat(feat_mean, dynamic.shape[2], axis=2)
            feat_mean = np.repeat(feat_mean, dynamic.shape[3], axis=3)
            dynamic = np.where(np.isnan(dynamic), feat_mean, dynamic)
      if self.nan_fill:
         dynamic = np.nan_to_num(dynamic, nan=self.nan_fill)
         static = np.nan_to_num(static, nan=self.nan_fill)

      if self.clc == 'mode':
         clc = np.load(str(path).replace('dynamic', 'clc_mode'))
      elif self.clc == 'vec':
         clc = np.load(str(path).replace('dynamic', 'clc_vec'))
         clc = np.nan_to_num(clc, nan=0)
      else:
         clc = 0
      path = Path(path)
      dirPath = Path(*path.parts[:-1])
      name = path.stem.split('_')[:-1]
      prefix = dirPath / '_'.join(name)
      return dynamic, static, clc, str(prefix)

   def __len__(self):
      return len(self.path_list)

class zigzagTDA:
   def __init__(self, alpha, NVertices, scaleParameter, maxDimHoles, sizeWindow):
      self.alpha = alpha
      self.NVertices = NVertices
      self.scaleParameter = scaleParameter
      self.maxDimHoles = maxDimHoles
      self.sizeWindow = sizeWindow

   #X: T, N, F
   def zigzag_persistence_diagrams(self, x):
       GraphsNetX = []
       for t in range(self.sizeWindow): 
         graphL2 = np.sum((x[t,np.newaxis, :, :]-x[t,:,np.newaxis,:])**2, axis=-1) #compute matrix distance
        # graphL2[graphL2==0] = 1e-5  ##weakly connected
         tmp_max = np.max(graphL2)
         graphL2 /= tmp_max  ##normalize  matrix
         graphL2[graphL2>self.alpha]=0 ##cut off note: probably this is not required
         GraphsNetX.append(graphL2)
   
       start_time = time.time()
      # Building unions and computing distance matrices
       print("Building unions and computing distance matrices...")  # Beginning
       MDisGUnions = []
       for i in range(0, self.sizeWindow - 1):
           # --- To concatenate graphs
           unionAux = []
           MDisAux = np.zeros((2 * self.NVertices, 2 * self.NVertices))
           A = GraphsNetX[i] #nx.adjacency_matrix(GraphsNetX[i]).todense()
           B = GraphsNetX[i+1] #nx.adjacency_matrix(GraphsNetX[i + 1]).todense()
           # ----- Version Original (2)
           C = (A + B) / 2
           A[A == 0] = 1.1
           A[range(self.NVertices), range(self.NVertices)] = 0
           B[B == 0] = 1.1
           B[range(self.NVertices), range(self.NVertices)] = 0
           MDisAux[0:self.NVertices, 0:self.NVertices] = A
           C[C == 0] = 1.1
           C[range(self.NVertices), range(self.NVertices)] = 0
           MDisAux[self.NVertices:(2 * self.NVertices), self.NVertices:(2 * self.NVertices)] = B
           MDisAux[0:self.NVertices, self.NVertices:(2 * self.NVertices)] = C
           MDisAux[self.NVertices:(2 * self.NVertices), 0:self.NVertices] = C.transpose()
           # Distance in condensed form
           pDisAux = squareform(MDisAux)
           # --- To save unions and distances
           MDisGUnions.append(pDisAux)  # To save distance matrix
       print("  --- End unions...")  # Ending
   
       # To perform Ripser computations
       print("Computing Vietoris-Rips complexes...")  # Beginning
   
       GVRips = []
       for jj in range(self.sizeWindow - 1):
           print(jj)
           ripsAux = d.fill_rips(MDisGUnions[jj], self.maxDimHoles, self.scaleParameter)
           GVRips.append(ripsAux)
       print("  --- End Vietoris-Rips computation")  # Ending
   
       # Shifting filtrations...
       print("Shifting filtrations...")  # Beginning
       GVRips_shift = []
       GVRips_shift.append(GVRips[0])  # Shift 0... original rips01
       for kk in range(1, self.sizeWindow - 1):
           shiftAux = zzt.shift_filtration(GVRips[kk], self.NVertices * kk)
           GVRips_shift.append(shiftAux)
       print("  --- End shifting...")  # Ending
   
       # To Combine complexes
       print("Combining complexes...")  # Beginning
       completeGVRips = zzt.complex_union(GVRips[0], GVRips_shift[1])
       for uu in range(2, self.sizeWindow - 1):
           completeGVRips = zzt.complex_union(completeGVRips, GVRips_shift[uu])
       print("  --- End combining")  # Ending
   
       # To compute the time intervals of simplices
       print("Determining time intervals...")  # Beginning
       time_intervals = zzt.build_zigzag_times(completeGVRips, self.NVertices, self.sizeWindow)
       print("  --- End time")  # Beginning
   
       # To compute Zigzag persistence
       print("Computing Zigzag homology...")  # Beginning
       G_zz, G_dgms, G_cells = d.zigzag_homology_persistence(completeGVRips, time_intervals)
       print("  --- End Zigzag")  # Beginning
   
       # To show persistence intervals
       window_ZPD = []
       # Personalized plot
       for vv, dgm in enumerate(G_dgms):
           print("Dimension:", vv)
           if (vv < 2):
               matBarcode = np.zeros((len(dgm), 2))
               k = 0
               for p in dgm:
                   matBarcode[k, 0] = p.birth
                   matBarcode[k, 1] = p.death
                   k = k + 1
               matBarcode = matBarcode / 2
               window_ZPD.append(matBarcode)
   
       # Timing
       print("TIME: " + str((time.time() - start_time)) + " Seg ---  " + str((time.time() - start_time) / 60) + " Min ---  " + str((time.time() - start_time) / (60 * 60)) + " Hr ")
   
       return window_ZPD
   
   # Zigzag persistence image
   def zigzag_persistence_images(self, dgms, resolution = [50,50], return_raw = False, normalization = True, bandwidth = 1., power = 1., dimensional = 0):
       PXs, PYs = np.vstack([dgm[:, 0:1] for dgm in dgms]), np.vstack([dgm[:, 1:2] for dgm in dgms])
       print("number selectors..", PXs.shape)
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
           norm_output = (output - np.min(output))/(np.max(output) - np.min(output))
       else:
           norm_output = output
   #    plt.plot(PXs, PYs, 'ro')
       
       print(norm_output.shape)
   
       fig, (ax1, ax2) = plt.subplots(1,2)
       fig.suptitle('TDA rules! '+str(dimensional)+'-dimensional ZPD')
       ax1.set_xlim(0, window-1)
       ax1.set_ylim(0, window-1)
       ax1.set_xlabel('birth')
       ax1.set_ylabel('death')
       ax1.plot(PXs, PYs, 'ro')
   
       X, Y = np.meshgrid(x, y)
       ax2.set_xlim(xm, xM)
       ax2.set_ylim(ym, yM)
       ax2.contourf(X, Y, norm_output)
       plt.savefig('my_plot'+str(dimensional)+'.pdf')
       plt.show()
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
#####TDA parameters
maxDimHoles = 1
window = 10
alpha = 1.
scaleParameter =  0.3
sizeBorder = 5
NVertices = (2*sizeBorder+1)**2

#data = np.random.rand(window, NVertices, 25)

###We can ge more plots from here...
###zzgraph.plotting(data, NVertices, alpha, scaleParameter, maxDimHoles, window)
### for each sample....

data =  loadData(dataset_root = dataset_root,access_mode = access_mode, dynamic_features = dynamic_features, static_features = static_features, clc=clc)
ZZ = zigzagTDA(alpha=alpha, NVertices=NVertices, scaleParameter=scaleParameter, maxDimHoles=maxDimHoles, sizeWindow=window)
print("Number of samples", len(data))
print("Dynamic shape..", data[0][0].shape)
print("Static shape..", data[0][1].shape)
print("clc shape..", data[0][2].shape)
for (dynamic, static, clc, prefix_path) in data:
   (sizeWindow, _ , patchWidth, patchHeight) = dynamic.shape
   numberFeatures = len(dynamic_features)+len(static_features)+len(clc)
   sample = np.zeros((sizeWindow, NVertices, numberFeatures))
   for t in range(sizeWindow):
      X = np.concatenate((dynamic[t], static, clc), axis=0) ##F, W, H
      X = X[:,12-sizeBorder:13+sizeBorder,12-sizeBorder:13+sizeBorder]
      X = X.reshape(numberFeatures, -1) # F, N
      sample[t] = X.transpose(1,0) #N, F
   print(prefix_path)
   zigzag_PD = ZZ.zigzag_persistence_diagrams(x = sample)
   print(len(zigzag_PD))
   zigzag_PI_H0 = ZZ.zigzag_persistence_images(zigzag_PD, dimensional = 0)
   zigzag_PI_H1 = ZZ.zigzag_persistence_images(zigzag_PD, dimensional = 1)
   print(zigzag_PI_H0)
   print(zigzag_PI_H1)
#
