# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:14:25 2024

@author: rpons
"""

import pickle
import numpy as np
import pandas as pd
import ruptures as rpt
from pathlib import Path
from scipy.io import loadmat
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import math
import mat73
from read_roi import read_roi_file
from shapely.geometry import Point, Polygon
from shapely import geometry
from scipy.optimize import curve_fit

from step.data import *
from step.models import *
#from step.utils import diffusion_coefficient_tmsd

from tqdm.auto import tqdm
from fastai.vision.all import *

file_paths_pre = tuple()

inp = int(input('How many conditions to compare? '))
n_conditions = []
conditions_labels = []
for i in range(inp):
    conditions_labels.append(input(f'Name condition {i+1}? '))
    n_conditions.append(int(input(f'How many files per condition {i+1}? ')))
    
#%%
"""Rerun the code from to select multiple files, load them in order (i.e. track(condition 1, 2, ... N), ROI(condition 1, 2, ... N))"""
root = tk.Tk()
root.withdraw()
root.call('wm', 'attributes', '.', '-topmost', True)

file_paths_pre = file_paths_pre + filedialog.askopenfilenames()

file_paths_pre_append = []
for j in range(sum(n_conditions)):
    file_paths_pre_append.append([file_paths_pre[j] , file_paths_pre[j+sum(n_conditions)]])

file_paths = []
for i in range(len(n_conditions)):    
    file_paths.append(file_paths_pre_append[sum(n_conditions[0:i]) : sum(n_conditions[0:i+1])])


#%%
frame_rate = 0.015 #seconds 0.0165
pixel_size = 0.15669 #um/pixel
trunc = 0.3 #fraction of trajectory to analyze

def MSD(x, y):
    msd = np.zeros(len(x)-1)
    for i in range(len(x)-1):
        msd[i] = sum(np.square(np.sqrt(np.square(x[i:]-x[:len(x)-i])+np.square(y[i:]-y[:len(x)-i]))))/(len(x)-i)
    return msd

def MSD_fit(x, D, a):
    y = 2*2*D*x**a
    return y

def inst_MSD_fit(x, D):
    y = 2*2*D*x
    return y

# Defino esto aquí para que no tengas que instalar la librería
def tamsd(x, dt=1):
    "Computes the time averaged mean squared displacement of a trajectory `x`."
    return ((x[dt:] - x[:-dt])**2).sum(-1).mean()

def diffusion_coefficient_tamsd(x, t_lag=[1, 2]):
    "Estimates the diffusion coefficient fitting the `tmsd` for different `dt`."
    tamsds = [tamsd(x, dt) for dt in t_lag]
    D = np.polyfit(t_lag, tamsds, 1)[0]
    return D/2/x.shape[-1]

def get_angle(a, b, c):
    vector_1 = b - a
    vector_2 = c - b 
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    ang = math.degrees(np.arccos(dot_product))
    d = np.cross(vector_2,vector_1)
    return ang if d < 0 else -ang

def dataset_angles(trajs):
    angles = []
    for traj in trajs:
        for a, b, c in zip(traj[:, :-2].transpose(), traj[:, 1:-1].transpose(), traj[:, 2:].transpose()):
            angles.append(get_angle(a, b, c))
    return angles

def confinement_radius(x, nm_per_pxl=160):
    """Compute the confinement radius of a trajectory segment."""
    cov = np.cov(x)
    e_vals, _ = np.linalg.eigh(cov)
    return 2 * nm_per_pxl * np.sqrt(e_vals.mean())

def get_roi_polygon(roi):
    roii = next(iter(roi))
    roiii = roi[roii]
    if roiii['type'] == 'rectangle':
        polygon_coordinates = np.array([(roiii['left'], roiii['top']), (roiii['left']+roiii['width'], roiii['top']), (roiii['left']+roiii['width'], roiii['top']+roiii['height']), (roiii['left'], roiii['top']+roiii['height'])])
    if roiii['type'] == 'polygon':
        polygon_coordinates = np.concatenate((np.array([roiii['x']]), np.array([roiii['y']]))).T
    polygon = Polygon(polygon_coordinates)
    #xt, yt = polygon.exterior.xy
    polygon = polygon.buffer(-2)
    #xtt, ytt = polygon.exterior.xy
    return polygon

def is_point_inside_polygon(x, y, polygon_coords):
    point = Point(x, y)
    polygon = Polygon(polygon_coords)
    return polygon.contains(point)

def filter_out_localizations_on_the_edge(x, y, frame):
    traj_is_inside = np.empty((0,1))
    max_length = 0
    max_start = np.empty((0,1))
    current_start = np.empty((0,1))
    current_length = 0
    for i in range(len(x)):
        is_inside = is_point_inside_polygon(x[i]/pixel_size, y[i]/pixel_size, polygon)
        traj_is_inside = np.vstack([traj_is_inside, is_inside]) #true=1, false=0
        if is_inside==True:
            if current_length == 0:
                current_start = i
            current_length += 1
            if current_length>max_length:
                max_start = current_start
            max_length = max(max_length, current_length)
        else:
            current_length = 0
    if max_length == 0:
        x = []
        y = []
        frame = []
    else:
        x = x[max_start: max_start+max_length]
        y = y[max_start: max_start+max_length]
        frame = frame[max_start: max_start+max_length]
    return x, y, frame

def predict(model, x):
    # return to_detach(model(x.cuda().T.unsqueeze(0).float()).squeeze())
    return to_detach(model(x.T.unsqueeze(0).float()).squeeze())

model_dir = "../../models"
dim = 2

dls = DataLoaders.from_dsets([], []) # Empty train and validation datasets

# Diffusion coefficient
# model_diff = XResAttn(dim, n_class=1, stem_szs=(64,), conv_blocks=[1, 1, 1], block_szs=[128, 256, 512], pos_enc=False,
#                  n_encoder_layers=4, dim_ff=512, nhead_enc=8, linear_layers=[], norm=False, yrange=(-3.1, 3.1))
# model_diff = XResAttn(dim, n_class=1, stem_szs=(64,), conv_blocks=[1, 1, 1], block_szs=[128, 256, 512], pos_enc=False,
#                  n_encoder_layers=4, dim_ff=512, nhead_enc=8, linear_layers=[], norm=False, yrange=(-4.1, 4.1))
model_diff = LogXResAttn(dim, n_class=1, stem_szs=(64,), conv_blocks=[1, 1, 1], block_szs=[128, 256, 512], pos_enc=False,
                         n_encoder_layers=4, dim_ff=512, nhead_enc=8, linear_layers=[], norm=False, yrange=(-7.1, 2.1))
model_diff.to(default_device())

# Anomalous exponent
model_exp = XResAttn(dim, n_class=1, stem_szs=(32,), conv_blocks=[1, 1, 1], block_szs=[128, 256, 512], 
                     pos_enc=False, n_encoder_layers=4, dim_ff=512, nhead_enc=8, linear_layers=[])
model_exp.to(default_device())

# Create the learners
learn_diff = Learner(dls, model_diff, loss_func=L1LossFlat(), model_dir=model_dir)
learn_exp = Learner(dls, model_exp, loss_func=L1LossFlat(), model_dir=model_dir)

learn_diff.load(fr'C:\Users\rpons\Documents\Python\STEP\models\logxresattn_bm_{dim}d_1_to_4_cp_juan_72')
learn_diff.model.eval();

learn_exp.load(fr'C:\Users\rpons\Documents\Python\STEP\models\xresattn_exp_{dim}d_no_pe_2')
learn_exp.model.eval();

list_parameters_t_msd = []
list_parameters_t_step_pred = []
for i in range(len(n_conditions)):
    list_parameters_cond_msd = []
    list_parameters_cond_step_pred = []
    for j in range(n_conditions[i]):
        print(j)
        track = pd.read_csv(file_paths[i][j][0], skiprows=[1,2,3], dtype={'first_row': 'str'})
        roi = read_roi_file(file_paths[i][j][1])
        
        x_all = (track.loc[:,"POSITION_X"]).to_numpy()
        y_all = (track.loc[:,"POSITION_Y"]).to_numpy()
        t_all = (track.loc[:,"POSITION_T"]).to_numpy()
        frame_all = (track.loc[:,"FRAME"]).to_numpy()+1 #!!! this +1 takes into account that the tracking for some weird reason has frames-1
        track_id = (track.loc[:,"TRACK_ID"]).to_numpy()
        
        list_parameters_msd = np.empty((0,2))
        list_parameters_step_pred = []
        polygon = get_roi_polygon(roi)
        
        for k in set(track_id):
            x = x_all[track_id==k]
            y = y_all[track_id==k]
            frame = frame_all[track_id==k]
            
            x, y, frame = filter_out_localizations_on_the_edge(x, y, frame)
            
            if len(x) >= 30:
                x_tnsr = torch.tensor(x/pixel_size)
                y_tnsr = torch.tensor(y/pixel_size)
                traj = torch.stack((x_tnsr-x_tnsr[0], y_tnsr-y_tnsr[0]))
                
                step_dif_pred = predict(learn_diff.model, traj).numpy()
                step_a_pred = predict(learn_exp.model, traj).numpy()
                
                msd = MSD(x,y)
                t = np.arange(0,int(len(msd)))*frame_rate
                inst_parameters, inst_covariance = curve_fit(inst_MSD_fit, t[0:4], msd[0:4])
                parameters, covariance = curve_fit(MSD_fit, t[0:round(len(x)*trunc)], msd[0:round(len(x)*trunc)])
                D_inst = inst_parameters[0]
                a = parameters[1]
                
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
                
                fig.suptitle(f'k = {k}, frame = {int(frame[0])} \n x = {int(x[0])}, y = {int(y[0])}') 
                ax1.plot(x[1:], -y[1:], color='black', zorder=1)
                cm1 = plt.cm.get_cmap('RdYlBu')
                sc1 = ax1.scatter(x[1:], -y[1:], c=step_dif_pred, s=35, cmap=cm1, zorder=2, vmin=-4, vmax=0)
                cbar1 = plt.colorbar(sc1, ax=ax1)
                cbar1.set_label('D coef')
                ax1.set_aspect('equal', adjustable='box')
                
                ax2.plot(x[:], -y[:], color='black', zorder=1)
                cm2 = plt.cm.get_cmap('RdYlBu')
                sc2 = ax2.scatter(x[:], -y[:], c=step_a_pred, s=35, cmap=cm1, zorder=2, vmin=0, vmax=2)
                cbar2 = plt.colorbar(sc2, ax=ax2)
                cbar2.set_label('alpha')
                ax2.set_aspect('equal', adjustable='box')
                
                
                list_parameters_msd = np.vstack([list_parameters_msd, np.vstack([[D_inst],[a]]).T])
                list_parameters_step_pred.append([[step_dif_pred],[step_a_pred]])
                
        #pdb.set_trace()
        mask = np.isnan(list_parameters_msd).any(axis=1)
        list_parameters_msd = list_parameters_msd[~mask]
        list_parameters_cond_msd.append(list_parameters_msd)
        
        list_parameters_step_pred = list_parameters_step_pred[~mask]
        list_parameters_cond_step_pred.append(list_parameters_step_pred)
        
    list_parameters_t_msd.append(list_parameters_cond_msd)
    list_parameters_t_step_pred.append(list_parameters_cond_step_pred)

plt.figure()
plt.hist(torch.cat(preds).numpy(), bins=15);

plt.figure()
preds_mean = [p.mean().numpy().item() for p in preds]
plt.hist(preds_mean, bins=15);

D_biot = np.concatenate(list_parameters_t_msd[0], axis=0)[:,0]
plt.figure()
plt.hist(np.log10(D_biot))

plt.figure()
plt.hist2d(preds_mean, np.log10(D_biot))

