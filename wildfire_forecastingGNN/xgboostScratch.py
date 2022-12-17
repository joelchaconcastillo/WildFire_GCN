import torch
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
import json
import warnings
from pathlib import Path

import random
from datetime import datetime

tmpseed = int(datetime.now().timestamp())%1000
print("MARK LINE")
print(tmpseed)
random.seed(tmpseed)

from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.metrics import RocCurveDisplay
from sklearn import metrics


from wildfire_forecasting.datamodules.datasets.greecefire_dataset import FireDataset_npy

torch.multiprocessing.set_sharing_strategy('file_system')

sel_dynamic_features = [
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


sel_static_features = [
 'dem_mean',
# 'aspect_mean',
 'slope_mean',
# 'roughness_mean',
 'roads_distance',
 'waterway_distance',
 'population_density',
]

clc = 'vec'


# !IMPORTANT fill the path with path of the dataset you have downloaded
dataset_root = Path("/home/joel.chacon/tmp/datasets_grl")

dataloaders = {
    'train' : torch.utils.data.DataLoader(FireDataset_npy(dataset_root=dataset_root, train_val_test='train', access_mode='temporal', static_features=sel_static_features, dynamic_features=sel_dynamic_features, clc = clc), batch_size=1, shuffle=True, num_workers=12),
    'val' : torch.utils.data.DataLoader(FireDataset_npy(dataset_root = dataset_root, train_val_test='val', access_mode = 'temporal', static_features=sel_static_features, dynamic_features=sel_dynamic_features, clc = clc), batch_size=1, num_workers=12),
    'test': torch.utils.data.DataLoader(FireDataset_npy(dataset_root = dataset_root, train_val_test='test', access_mode = 'temporal', static_features=sel_static_features, dynamic_features=sel_dynamic_features, clc = clc), batch_size=1, num_workers=12),
}


#Create the training, val and test datasets
X_train = []
X_val = []
X_test = []
y_train = []
y_val = []
y_test = []

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    for i, (dynamic, static, clc, label) in enumerate(dataloaders['train']):
        dynamic_avg = torch.from_numpy(np.nanmean(dynamic.numpy(), axis=1))
        input_ = torch.cat([dynamic_avg.squeeze(), dynamic[:,-1,:].squeeze(), static.squeeze(), clc.squeeze()], dim = 0)
        input_ = input_.numpy()
        X_train.append(input_)
        y_train.append(label.numpy())

    for i, (dynamic, static, clc, label) in enumerate(dataloaders['val']):
        dynamic_avg = torch.from_numpy(np.nanmean(dynamic.numpy(), axis=1))
        input_ = torch.cat([dynamic_avg.squeeze(), dynamic[:,-1,:].squeeze(), static.squeeze(), clc.squeeze()], dim = 0)
        input_ = input_.numpy()
        X_val.append(input_)
        y_val.append(label.numpy())

    for i, (dynamic, static, clc, label) in enumerate(dataloaders['test']):
        dynamic_avg = torch.from_numpy(np.nanmean(dynamic.numpy(), axis=1))
        input_ = torch.cat([dynamic_avg.squeeze(), dynamic[:,-1,:].squeeze(), static.squeeze(), clc.squeeze()], dim = 0)
        input_ = input_.numpy()
        X_test.append(input_)
        y_test.append(label.numpy())

X_train = np.stack(X_train, axis=0)
y_train = np.stack(y_train, axis=0)
X_val = np.stack(X_val, axis=0)
y_val = np.stack(y_val, axis=0)
X_test = np.stack(X_test, axis=0)
y_test = np.stack(y_test, axis=0)


num_trees= 500
max_depth = 1
min_split_loss = 0
learning_rate = 0.3
min_sum_inst_weight_child = 2
subsample_ratio_each_tree = 0.5
#xgb_model = xgb.XGBRFClassifier(n_estimators = num_trees, random_state=seed, max_depth=3)
xgb_model = XGBClassifier(n_estimators = num_trees, max_depth=max_depth, gamma=min_split_loss , learning_rate=learning_rate, min_child_weight=min_sum_inst_weight_child, subsample=subsample_ratio_each_tree, random_state=tmpseed)

xgb_model.fit(X_train, y_train.ravel(), eval_set = [(X_train, y_train.ravel()),(X_val, y_val.ravel())] )
#xgb_model.fit(X_train, y_train.ravel(), eval_set = [(X_train, y_train.ravel())] )


y_pred=xgb_model.predict(X_test)

probs_pred = xgb_model.predict_proba(X_test)[:,1]
X_test = np.stack(X_test, axis=0)
y_test = np.stack(y_test, axis=0)
auc = roc_auc_score(y_test, probs_pred)
aucpr = average_precision_score(y_test, probs_pred)

print(auc)
print(aucpr)
print(classification_report(y_test, y_pred, digits=3))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#print(tp, fp, tn, fn)
summary = classification_report(y_test, y_pred, digits=3, output_dict=True)['1']
summary['AUC']=auc
summary['AUCPR']=aucpr
summary['TP']=tp
summary['FP']=fp
summary['TN']=tn
summary['FN']=fn
print(summary)

