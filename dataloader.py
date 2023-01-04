import torch
import warnings
import numpy as np
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, TensorDataset
import random
from datetime import datetime
import json
import random
import shutil
from pathlib import Path
from ZPIcreator import zigzagTDA


class FireDSDataModuleGCN:
    """
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
            self, args):
        super().__init__()
        self.args = args

        if not args.dataset_root:
            raise ValueError('dataset_root variable must be set. Check README')
        self.ZZ = zigzagTDA(alpha=args.alpha, scaleParameter=args.scaleParameter, maxDimHoles=args.maxDimHoles, sizeWindow=args.lag, sizeBorder = args.sizeBorder)

        self.data_train = FireDataset_Graph_npy(dataset_root=args.dataset_root,
                                          train_val_test='train',
                                          dynamic_features=args.dynamic_features,
                                          static_features=args.static_features,
                                          nan_fill=args.nan_fill, clc=args.clc, ZZ=self.ZZ)
        self.data_val = FireDataset_Graph_npy(dataset_root=args.dataset_root, 
                                        train_val_test='val',
                                        dynamic_features=args.dynamic_features,
                                        static_features=args.static_features,
                                        nan_fill=args.nan_fill, clc=args.clc, ZZ=self.ZZ)
        self.data_test1 = FireDataset_Graph_npy(dataset_root=args.dataset_root,
                                         train_val_test='test1',
                                         dynamic_features=args.dynamic_features,
                                         static_features=args.static_features,
                                         nan_fill=args.nan_fill, clc=args.clc, ZZ=self.ZZ)
        self.data_test2 = FireDataset_Graph_npy(dataset_root=args.dataset_root,
                                         train_val_test='test2',
                                         dynamic_features=args.dynamic_features,
                                         static_features=args.static_features,
                                         nan_fill=args.nan_fill, clc=args.clc, ZZ=self.ZZ)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.args.minbatch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=True,
            prefetch_factor=self.args.prefetch_factor,
            persistent_workers=self.args.persistent_workers
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.args.minbatch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=False,
            prefetch_factor=self.args.prefetch_factor,
            persistent_workers=self.args.persistent_workers
        )

    def test_dataloader1(self):
        return DataLoader(
            dataset=self.data_test1,
            batch_size=self.args.minbatch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=False,
            prefetch_factor=self.args.prefetch_factor,
            persistent_workers=self.args.persistent_workers
        )
    def test_dataloader2(self):
        return DataLoader(
            dataset=self.data_test2,
            batch_size=self.args.minbatch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=False,
            prefetch_factor=self.args.prefetch_factor,
            persistent_workers=self.args.persistent_workers
        )



class FireDataset_Graph_npy(Dataset):
    def __init__(self, dataset_root: str = None, access_mode: str = 'spatiotemporal',   
                 problem_class: str = 'classification',
                 train_val_test: str = 'train', dynamic_features: list = None, static_features: list = None,
                 categorical_features: list = None, nan_fill: float = -1., neg_pos_ratio: int = 2, clc: str = None, ZZ = None):
        """
        @param dataset_root: str where the dataset resides. It must contain also the minmax_clc.json
                and the variable_dict.json
        @param access_mode: spatial, temporal or spatiotemporal
        @param problem_class: classification or segmentation
        @param train_val_test:
                'train' gets samples from [2009-2018].
                'val' gets samples from 2019.
                test' get samples from 2020
        @param dynamic_features: selects the dynamic features to return
        @param static_features: selects the static features to return
        @param categorical_features: selects the categorical features
        @param nan_fill: Fills nan with the value specified here
        """
        # dataset_root should be a str leading to the path where the data have been downloaded and decompressed
        # Make sure to follow the details in the readme for that
        if not dataset_root:
            raise ValueError('dataset_root variable must be set. Check README')
        dataset_root = Path(dataset_root)
        min_max_file = dataset_root / 'minmax_clc.json'
        variable_file = dataset_root / 'variable_dict.json'
        with open(min_max_file) as f:
            self.min_max_dict = json.load(f)

        with open(variable_file) as f:
            self.variable_dict = json.load(f)

        self.static_features = static_features
        self.dynamic_features = dynamic_features
        self.nan_fill = nan_fill
        self.clc = clc
        self.access_mode = 'spatiotemporal'
        self.ZZ = ZZ
        dataset_path = dataset_root / 'npy' / self.access_mode
        self.positives_list = list((dataset_path / 'positives').glob('*dynamic.npy'))
        self.positives_list = list(zip(self.positives_list, [1] * (len(self.positives_list))))
        val_year = 2019
        test_year1 = 2020 #min(val_year + 1, 2021)
        test_year2 = 2021 #min(val_year + 1, 2021)

#        self.train_positive_list = [(x, y) for (x, y) in self.positives_list if int(x.stem[:4]) < 2012]
        self.train_positive_list = [(x, y) for (x, y) in self.positives_list if int(x.stem[:4]) < val_year]
        self.val_positive_list = [(x, y) for (x, y) in self.positives_list if int(x.stem[:4]) == val_year]
        self.test1_positive_list = [(x, y) for (x, y) in self.positives_list if int(x.stem[:4]) == test_year1]
        self.test2_positive_list = [(x, y) for (x, y) in self.positives_list if int(x.stem[:4]) == test_year2]

        self.negatives_list = list((dataset_path / 'negatives_clc').glob('*dynamic.npy'))
        self.negatives_list = list(zip(self.negatives_list, [0] * (len(self.negatives_list))))

        self.train_negative_list = random.sample(
            [(x, y) for (x, y) in self.negatives_list if int(x.stem[:4]) < val_year],
            len(self.train_positive_list) * neg_pos_ratio)
        self.val_negative_list = random.sample(
            [(x, y) for (x, y) in self.negatives_list if int(x.stem[:4]) == val_year],
            len(self.val_positive_list) * neg_pos_ratio)
        self.test1_negative_list = random.sample(
            [(x, y) for (x, y) in self.negatives_list if int(x.stem[:4]) == test_year1],
            len(self.test1_positive_list) * neg_pos_ratio)
        self.test2_negative_list = random.sample(
            [(x, y) for (x, y) in self.negatives_list if int(x.stem[:4]) == test_year2],
            len(self.test2_positive_list) * neg_pos_ratio)


        self.dynamic_idxfeat = [(i, feat) for i, feat in enumerate(self.variable_dict['dynamic']) if
                                feat in self.dynamic_features]
        self.static_idxfeat = [(i, feat) for i, feat in enumerate(self.variable_dict['static']) if
                               feat in self.static_features]
        self.dynamic_idx = [x for (x, _) in self.dynamic_idxfeat]
        self.static_idx = [x for (x, _) in self.static_idxfeat]

        if train_val_test == 'train':
            print(f'Positives: {len(self.train_positive_list)} / Negatives: {len(self.train_negative_list)}')
            self.path_list = self.train_positive_list + self.train_negative_list
        elif train_val_test == 'val':
            print(f'Positives: {len(self.val_positive_list)} / Negatives: {len(self.val_negative_list)}')
            self.path_list = self.val_positive_list + self.val_negative_list
        elif train_val_test == 'test1':
            print(f'Positives: {len(self.test1_positive_list)} / Negatives: {len(self.test1_negative_list)}')
            self.path_list = self.test1_positive_list + self.test1_negative_list
        elif train_val_test == 'test2':
            print(f'Positives: {len(self.test2_positive_list)} / Negatives: {len(self.test2_negative_list)}')
            self.path_list = self.test2_positive_list + self.test2_negative_list

        print("Dataset length", len(self.path_list))
        random.shuffle(self.path_list)
        self.mm_dict = self._min_max_vec()

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

    def _min_max_vec(self):
        mm_dict = {'min': {}, 'max': {}}
        for agg in ['min', 'max']:
           mm_dict[agg]['dynamic'] = np.ones((1, len(self.dynamic_features), 1, 1))
           mm_dict[agg]['static'] = np.ones((len(self.static_features), 1, 1))
           for i, (_, feat) in enumerate(self.dynamic_idxfeat):
               mm_dict[agg]['dynamic'][:, i, :, :] = self.min_max_dict[agg][self.access_mode][feat]
           for i, (_, feat) in enumerate(self.static_idxfeat):
               mm_dict[agg]['static'][i, :, :] = self.min_max_dict[agg][self.access_mode][feat]
        return mm_dict

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        path, labels = self.path_list[idx]
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
        _, W, H = clc.shape
        data = self.combine_dynamic_static_inputs(dynamic, static, clc)
        return data, labels, self.ZPIcreation(data, W, H)

    def ZPIcreation(self, data, W, H):
       '''
          data: T,F,N
       '''
       T, F, N = data.shape
       sample = data.reshape(T, F, W, H)
       sample = sample[:,:,12-self.ZZ.sizeBorder:13+self.ZZ.sizeBorder,12-self.ZZ.sizeBorder:13+self.ZZ.sizeBorder]
       sample = sample.reshape(T, F, -1) # T, F, N
       sample = sample.transpose(0,2,1) #T, N, F
       zigzag_PD = self.ZZ.zigzag_persistence_diagrams(x = sample)
       zigzag_PI_H0 = self.ZZ.zigzag_persistence_images(zigzag_PD, dimensional = 0)
       zigzag_PI_H1 = self.ZZ.zigzag_persistence_images(zigzag_PD, dimensional = 1)
       return [zigzag_PI_H0, zigzag_PI_H1]
   
def get_dataloaders(args):

  dataFireModule = FireDSDataModuleGCN(args)
  train_dataloader = dataFireModule.train_dataloader()

  val_dataloader = dataFireModule.val_dataloader()

  test_dataloader1 = dataFireModule.test_dataloader1()
  test_dataloader2 = dataFireModule.test_dataloader2()

#  print('Train: x y ->', x_tra.shape, topo_tra.shape, y_tra.shape)
#  print('Val: x, y ->', x_val.shape, topo_val.shape, y_val.shape)
#  print('Test: x, ZPI, y ->', x_test.shape, topo_test.shape, y_test.shape)
   ######################get triple dataloader######################

  return train_dataloader, val_dataloader, test_dataloader1, test_dataloader2
