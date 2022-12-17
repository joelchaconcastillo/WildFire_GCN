from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import warnings
import json
import scipy.sparse
from pathlib import Path

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

dataset_root = "/home/jcc/wildfire_forecastingGNN/data/datasets_grl"

alpha = 0.3

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
class sieveGraph:
    def __init__(self, data : Dataset = None):
      self.data = data
      print("Number of samples", len(self.data))
      print("Dynamic shape..", self.data[0][0].shape)
      print("Static shape..", self.data[0][1].shape)
      print("clc shape..", self.data[0][2].shape)

    def sieve(self, alpha : float = 1.0):
      postfix = "_"+str(alpha)
      for (dynamic, static, clc, prefix) in self.data:
          (windowSize, numberFeatures, patchWidth, patchHeight) = dynamic.shape
          numberNodes = patchWidth*patchHeight
          graphWindow = np.zeros((windowSize, numberNodes, numberNodes))
          for time in range(windowSize):
              X = np.concatenate((dynamic[time], static, clc), axis=0)
              (feat, width, height) = X.shape
              X = X.reshape(feat, -1)
              L2 = np.sum((X[:, np.newaxis, :]-X[:,:,np.newaxis])**2, axis=0) #compute matrix distance
              tmpMax = np.max(L2)
              L2 /=tmpMax
              L2[L2<1e-5] = 1e-5 ##handle numeric errors
              L2[L2 > alpha] = 0.0 ##is this necessary?
              graphWindow[time] = L2
          np.savez_compressed(prefix+"_graph", graph=graphWindow)
          print(prefix)

data =  loadData(dataset_root = dataset_root,access_mode = access_mode, dynamic_features = dynamic_features, static_features = static_features, clc=clc)

graphFire = sieveGraph(data)
graphFire.sieve(0.3)


