import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import os
import time
import random

from tqdm.notebook import tqdm

from . import network
from . import configs
import matplotlib.pyplot as plt


def get_not_in_tissue_coords(coords, img_xy):
    img_x, img_y = img_xy
    coords = coords.astype(img_x.dtype)
    coords = [list(val) for val in np.array(coords)]
    not_in_tissue_coords = []
    not_in_tissue_index = []
    for i in range(img_x.shape[0]):
        for j in range(img_x.shape[1]):
            ij_coord = [img_x[i, j], img_y[i, j]]
            if ij_coord not in coords:
                not_in_tissue_coords.append(ij_coord)
                not_in_tissue_index.append([int(i), int(j)])
    return not_in_tissue_coords, np.array(not_in_tissue_index).astype(np.int32)


def get10Xtrainset(train_counts, train_coords):
    time_start = time.time()
    train_counts = np.array(train_counts)
    train_coords = np.array(train_coords)
    delta_x = 1
    delta_y = 2

    # Get low-resolution images by downsampling.
    x_min = min(train_coords[:, 0]) + min(train_coords[:, 0]) % 2  # start with even row
    y_min = min(train_coords[:, 1]) + min(train_coords[:, 1]) % 2  # start with even col
    lr_x, lr_y = np.mgrid[x_min:max(train_coords[:, 0]) + delta_x:2 * delta_x,
                 y_min:max(train_coords[:, 1]):2 * delta_y]

    lr_spot_index = []
    lr_xy = [list(i) for i in list(np.vstack((lr_x.reshape(-1), lr_y.reshape(-1))).T)]
    for i in range(train_coords.shape[0]):
        if list(train_coords[i]) in lr_xy:
            lr_spot_index.append(i)
    lr_counts = train_counts[lr_spot_index]
    lr_coords = train_coords[lr_spot_index]

    # Get the coordinates of missing values on low-resolution images.
    lr_not_in_tissue_coords, lr_not_in_tissue_xy = get_not_in_tissue_coords(lr_coords, (lr_x, lr_y))
    lr_not_in_tissue_x = lr_not_in_tissue_xy.T[0]
    lr_not_in_tissue_y = lr_not_in_tissue_xy.T[1]

    train_lr = [None] * train_counts.shape[1]
    for i in range(train_counts.shape[1]):
        lr = griddata(lr_coords, lr_counts[:, i], (lr_x, lr_y), method="nearest")
        lr[lr_not_in_tissue_x, lr_not_in_tissue_y] = 0
        train_lr[i] = lr
    train_lr = np.array(train_lr)

    # Get high-resolution images as ground truths.
    hr_x, hr_y = np.mgrid[x_min:max(lr_coords[:, 0]) + delta_x * 1.5:delta_x,
                 y_min:max(lr_coords[:, 1]) + delta_y * 1.5:delta_y]

    for i in range(1, hr_y.shape[0], 2):
        hr_y[i] = hr_y[i] - delta_y / 2

    # Get the coordinates of missing values on high-resolution images.
    hr_not_in_tissue_coords, hr_not_in_tissue_xy = get_not_in_tissue_coords(train_coords, (hr_x, hr_y))
    hr_not_in_tissue_x = hr_not_in_tissue_xy.T[0]
    hr_not_in_tissue_y = hr_not_in_tissue_xy.T[1]

    train_hr = [None] * train_counts.shape[1]
    for i in tqdm(range(train_counts.shape[1])):
        hr = griddata(train_coords, train_counts[:, i], (hr_x, hr_y), method="nearest")
        hr[hr_not_in_tissue_x, hr_not_in_tissue_y] = 0
        train_hr[i] = hr
    train_hr = np.array(train_hr)

    in_tissue_matrix = np.ones_like(hr_x)
    in_tissue_matrix[hr_not_in_tissue_x, hr_not_in_tissue_y] = 0

    time_end = time.time()
    print('Creating training set costs: ', round(time_end - time_start, 2), 's')
    return train_lr, train_hr, in_tissue_matrix


def get10Xtestset(test_counts, test_coords):
    # Create test set from 10X Visium data or other spatial transcriptomic data having spots with hexagonal honeycomb-shaped arrangement.
    time_start = time.time()
    test_counts = np.array(test_counts)
    test_coords = np.array(test_coords)
    delta_x = 1
    delta_y = 2

    # Get coordinates of spots on testing images.
    x_min = min(test_coords[:, 0]) - min(test_coords[:, 0]) % 2  # start with even row
    y_min = min(test_coords[:, 1]) - min(test_coords[:, 1]) % 2  # start with even col

    test_input_x, test_input_y = np.mgrid[x_min:max(test_coords[:, 0]) + delta_x:delta_x,
                                 y_min:max(test_coords[:, 1]) + delta_y:delta_y]

    for i in range(1, test_input_y.shape[0], 2):
        test_input_y[i] = test_input_y[i] + delta_y / 2

    # Get the coordinates of missing values on input images.
    not_in_tissue_coords, not_in_tissue_xy = get_not_in_tissue_coords(test_coords, (test_input_x, test_input_y))
    not_in_tissue_x = not_in_tissue_xy.T[0]
    not_in_tissue_y = not_in_tissue_xy.T[1]

    # Get testing set.
    test_set = [None] * test_counts.shape[1]
    for i in tqdm(range(test_counts.shape[1])):
        test_data = griddata(test_coords, test_counts[:, i], (test_input_x, test_input_y), method="nearest")
        test_data[not_in_tissue_x, not_in_tissue_y] = 0
        test_set[i] = test_data
    test_set = np.array(test_set)

    time_end = time.time()
    print('Creating testing set costs: ', round(time_end - time_start, 2), 's')
    return test_set

import numpy as np
import pandas as pd
import time


def get_10X_position_info(integral_coords):

    time_start = time.time()
    integral_coords = np.array(integral_coords)
    delta_x = 1
    delta_y = 2

    # Get coordinates of imputed spots.
    x_min = min(integral_coords[:, 0]) - min(integral_coords[:, 0]) % 2  # start with even row
    y_min = min(integral_coords[:, 1]) - min(integral_coords[:, 1]) % 2  # start with even col

    y = list(np.arange(y_min, max(integral_coords[:, 1]) + delta_y, delta_y))
    imputed_x, imputed_y = np.mgrid[x_min:max(integral_coords[:, 0]) + delta_x:delta_x / 2,
                           y_min:y[-1] + delta_y:delta_y / 2]

    for i in range(1, imputed_y.shape[0], 2):
        imputed_y[i] -= delta_y / 4
    for i in range(2, imputed_y.shape[0], 4):
        imputed_y[i:i + 2] += delta_y / 2

    # Create mapping of imputed to nearest integral coordinates
    imputed_to_integral_mapping = {}
    for i in range(imputed_x.shape[0]):
        for j in range(imputed_x.shape[1]):
            imputed_coord = (imputed_x[i, j], imputed_y[i, j])
            distances = np.linalg.norm(integral_coords - imputed_coord, axis=1)
            nearest_integral_coord = integral_coords[np.argmin(distances)]
            imputed_to_integral_mapping[imputed_coord] = tuple(nearest_integral_coord)

    # Count number of adjacent original spots
    integral_coords = integral_coords.astype(np.float32)
    imputed_barcodes = [str(val[0]) + "x" + str(val[1]) for val in
                        np.vstack((imputed_x.reshape(-1), imputed_y.reshape(-1))).T]
    imputed_coords = pd.DataFrame(np.vstack((imputed_x.reshape(-1), imputed_y.reshape(-1))).astype(np.float32).T,
                                  columns=['row', 'col'], index=imputed_barcodes)
    neighbor_matrix = pd.DataFrame(np.zeros((imputed_coords.shape[0], imputed_coords.shape[0]), dtype=np.int32),
                                   columns=imputed_barcodes, index=imputed_barcodes)

    row1 = imputed_coords[imputed_coords["row"] == min(imputed_coords["row"])].sort_values("col")
    for i in range(len(row1) - 1):
        if row1["col"][i + 1] - row1["col"][i] == delta_y / 2:
            neighbor_matrix.loc[row1.index[i], row1.index[i + 1]] = 1
            neighbor_matrix.loc[row1.index[i + 1], row1.index[i]] = 1
    for row in tqdm(list(np.array(imputed_x).T[0])[:-1]):
        row0 = imputed_coords[imputed_coords["row"] == row].sort_values("col")
        row1 = imputed_coords[imputed_coords["row"] == row + delta_x / 2].sort_values("col")
        for i in range(len(row1) - 1):
            if row1["col"][i + 1] - row1["col"][i] == delta_y / 2:
                neighbor_matrix.loc[row1.index[i], row1.index[i + 1]] = 1
                neighbor_matrix.loc[row1.index[i + 1], row1.index[i]] = 1
        for i in range(len(row0)):
            for j in range(len(row1)):
                flag = 0
                if abs(imputed_coords.loc[row0.index[i], "col"] - imputed_coords.loc[
                    row1.index[j], "col"]) == delta_y / 4:
                    neighbor_matrix.loc[row0.index[i], row1.index[j]] = 1
                    neighbor_matrix.loc[row1.index[j], row0.index[i]] = 1
                    flag += 1
                if flag >= 2:
                    continue

    # Get not-in-tissue coordinates.
    neighbor_matrix = neighbor_matrix.loc[:, [str(val[0]) + "x" + str(val[1]) for val in integral_coords]]
    not_in_tissue_coords = []
    for i in range(len(imputed_coords)):
        if imputed_coords.index[i] in neighbor_matrix.columns:
            continue
        if sum(neighbor_matrix.iloc[i]) < 2:
            not_in_tissue_coords.append(list(imputed_coords.iloc[i]))

    position_info = [imputed_x, imputed_y, not_in_tissue_coords]

    return position_info, imputed_to_integral_mapping


def interpolation_10X(counts, coords, method = 'cubic'):

    counts = np.array(counts)
    coords = np.array(coords)   
    delta_x = 1
    delta_y = 2    
    x_min = min(coords[:,0]) - min(coords[:,0])%2 # start with even row
    y_min = min(coords[:,1]) - min(coords[:,1])%2 # start with even col    
    y=list(np.arange(y_min,max(coords[:,1])+delta_y,delta_y))
    interpolated_x, interpolated_y = np.mgrid[x_min:max(coords[:,0])+delta_x:delta_x/2, 
                                y_min:y[-1]+delta_y:delta_y/2]    
    for i in range(1,interpolated_y.shape[0],2):
        interpolated_y[i] -= delta_y/4
    for i in range(2,interpolated_y.shape[0],4):
        interpolated_y[i:i+2] += delta_y/2
        
    interpolated_set = [None] * counts.shape[1]
    for i in range(counts.shape[1]):
        interpolated_data = griddata(coords, counts[:,i], (interpolated_x, interpolated_y), method=method,fill_value=0)
        interpolated_set[i] = interpolated_data
    
    interpolated_set = np.array(interpolated_set)
    if np.sum(counts<0)==0:
        interpolated_set[interpolated_set<0] = 0    
    
    return interpolated_set


def getSTtrainset(train_counts, train_coords):

    time_start=time.time()
    train_counts = np.array(train_counts)
    train_coords = np.array(train_coords)
    train_coords[:,0] = train_coords[:,0] - min(train_coords[:,0])
    train_coords[:,1] = train_coords[:,1] - min(train_coords[:,1])   
    delta_x = 1
    delta_y = 1
      
    # The numbers of hr's row and columns must be even.
    if not max(train_coords[:,0])%2:
        x_index = (train_coords[:,0]<max(train_coords[:,0]))
        train_coords = train_coords[x_index]
        train_counts = train_counts[x_index]
    if not max(train_coords[:,1])%2:
        y_index = (train_coords[:,1]<max(train_coords[:,1]))
        train_coords = train_coords[y_index]
        train_counts = train_counts[y_index] 
    
    # Get low-resolution images by downsampling.
    lr_spot_index = []
    lr_x, lr_y=np.mgrid[0:max(train_coords[:,0]):2*delta_x,
                  0:max(train_coords[:,1]):2*delta_y] 
    
    lr_xy=[list(i) for i in list(np.vstack((lr_x.reshape(-1),lr_y.reshape(-1))).T)]
    for i in range(train_coords.shape[0]):
        if list(train_coords[i]) in lr_xy:
            lr_spot_index.append(i)    
    lr_counts = train_counts[lr_spot_index]
    lr_coords = train_coords[lr_spot_index]
    
    # Get the coordinates of missing values on low-resolution images.
    lr_not_in_tissue_coords, lr_not_in_tissue_xy=get_not_in_tissue_coords(lr_coords, (lr_x,lr_y))
    lr_not_in_tissue_x=lr_not_in_tissue_xy.T[0]
    lr_not_in_tissue_y=lr_not_in_tissue_xy.T[1] 
    
    train_lr = [None] * train_counts.shape[1]
    for i in range(train_counts.shape[1]):
        lr = griddata(lr_coords, lr_counts[:,i], (lr_x, lr_y),method="nearest")
        lr[lr_not_in_tissue_x,lr_not_in_tissue_y]=0
        train_lr[i]=lr         
    train_lr = np.array(train_lr)

    # Get high-resolution images as ground truths.
    hr_x ,hr_y = np.mgrid[0:max(train_coords[:,0])+delta_x:delta_x, 
                   0:max(train_coords[:,1])+delta_y:delta_y]  

    
    # Get the coordinates of missing values on high-resolution images.    
    hr_not_in_tissue_coords, hr_not_in_tissue_xy=get_not_in_tissue_coords(train_coords, (hr_x,hr_y))
    hr_not_in_tissue_x=hr_not_in_tissue_xy.T[0]
    hr_not_in_tissue_y=hr_not_in_tissue_xy.T[1]

    train_hr = [None] * train_counts.shape[1]
    for i in range(train_counts.shape[1]):        
        hr = griddata(train_coords, train_counts[:,i], (hr_x, hr_y),method="nearest")
        hr[hr_not_in_tissue_x,hr_not_in_tissue_y]=0
        train_hr[i] = hr
    train_hr = np.array(train_hr)
        
    in_tissue_matrix = np.ones_like(hr_x)
    in_tissue_matrix[hr_not_in_tissue_x, hr_not_in_tissue_y] = 0
    
    time_end=time.time()
    print('Creating training set costs: ',round(time_end-time_start,2),'s')
    return train_lr,train_hr,in_tissue_matrix

def getSTtestset(test_counts, test_coords):

    time_start=time.time()
    test_counts = np.array(test_counts)
    test_coords = np.array(test_coords)    
    delta_x = 1
    delta_y = 1
    
    # Get coordinates of spots on testing images.
    test_input_x, test_input_y = np.mgrid[min(test_coords[:,0]):max(test_coords[:,0])+delta_x:delta_x, 
                             min(test_coords[:,1]):max(test_coords[:,1])+delta_y:delta_y]    
        
    # Get the coordinates of missing values on input images.
    not_in_tissue_coords, not_in_tissue_xy=get_not_in_tissue_coords(test_coords, (test_input_x,test_input_y))
    not_in_tissue_x = not_in_tissue_xy.T[0]
    not_in_tissue_y = not_in_tissue_xy.T[1]
    
    # Get testing set.
    test_set = [None] * test_counts.shape[1]
    for i in range(test_counts.shape[1]):
        test_data = griddata(test_coords, test_counts[:,i], (test_input_x, test_input_y),method="nearest")
        test_data[not_in_tissue_x,not_in_tissue_y]=0
        test_set[i] = test_data        
    test_set = np.array(test_set)
        
    time_end=time.time()
    print('Creating testing set costs: ',round(time_end-time_start,2),'s')
    return test_set

# def get_ST_position_info(integral_coords):
#     """
#     Get information about imputed coordinates for judging weather an imputed spot is in tissue and preserved in the final result. The function is applicable to the situation that integral_coords is related to ST data or other spatial transcriptomic data having spots with matrix arrangement. It can be collocate with getSTtestset() or interpolation_ST().
#
#     Parameter:
#     intergral_coords: a 2-D ndarray or dataframe of spots coordinates. Each row corresponds to a spot with a barcode, the first column corresponds to row coordinates of the spots and the second column corresponds to column coordinates of the spots. Integral_coords describes row and column coordinates of some original spots that are expected to preserve. If an imputed spot points to an original spot or stays between two adjacent original spots from integral_coords, it will be preserved finally. Usually, intergral_coords contains the initial spots of test data without quality control for tissue structural integrity. You can add manually some spots coordinates which you want to preserve.
#
#     Return:
#     position_info: information about imputed coordinates.
#     position_info = [imputed_x, imputed_y, not_in_tissue_coords]
#     imputed_x: a 2-D ndarray. Row coordinates of imputed spots.
#     imputed_y: a 2-D ndarray with the same shape as imputed_x. Column coordinates of imputed spots.
#     not_in_tissue_coords: a list of two-element tuples. Row and column coordinates of imputed spots outsides tissue.
#     """
#     integral_coords = np.array(integral_coords)
#     delta_x = 1
#     delta_y = 1
#
#     # Get coordinates of imputed spots.
#     imputed_x, imputed_y = np.mgrid[min(integral_coords[:,0]):max(integral_coords[:,0])+delta_x:delta_x/2,
#                          min(integral_coords[:,1]):max(integral_coords[:,1])+delta_y:delta_y/2]
#
#     # Count the number of adjacent spots from original data.
#     integral_coords = integral_coords.astype(np.float32)
#     imputed_barcodes=[str(val[0])+"x"+str(val[1]) for val in np.vstack((imputed_x.reshape(-1),imputed_y.reshape(-1))).T]
#     imputed_coords = pd.DataFrame(np.vstack((imputed_x.reshape(-1),imputed_y.reshape(-1))).astype(np.float32).T,
#                                columns=['row','col'],index=imputed_barcodes)
#     neighbor_matrix=pd.DataFrame(np.zeros((imputed_coords.shape[0],imputed_coords.shape[0]),dtype=np.int32),
#                              columns=imputed_barcodes,index=imputed_barcodes)
#
#     row1=imputed_coords[imputed_coords["row"]==min(imputed_coords["row"])].sort_values("col")
#     for i in range(len(row1)-1):
#         if row1["col"][i+1]-row1["col"][i]==delta_y/2:
#             neighbor_matrix.loc[row1.index[i],row1.index[i+1]]=1
#             neighbor_matrix.loc[row1.index[i+1],row1.index[i]]=1
#     for row in list(np.array(imputed_x).T[0])[:-1]:
#         row0=imputed_coords[imputed_coords["row"]==row].sort_values("col")
#         row1=imputed_coords[imputed_coords["row"]==row+delta_x/2].sort_values("col")
#         for i in range(len(row1)-1):
#             if row1["col"][i+1]-row1["col"][i]==delta_y/2:
#                 neighbor_matrix.loc[row1.index[i],row1.index[i+1]]=1
#                 neighbor_matrix.loc[row1.index[i+1],row1.index[i]]=1
#         for i in range(len(row0)):
#             for j in range(len(row1)):
#                 flag=0
#                 if abs(imputed_coords.loc[row0.index[i],"col"]-imputed_coords.loc[row1.index[j],"col"])==delta_y/2:
#                     neighbor_matrix.loc[row0.index[i],row1.index[j]]= -1
#                     neighbor_matrix.loc[row1.index[j],row0.index[i]]= -1
#                     flag+=1
#                 if imputed_coords.loc[row0.index[i],"col"]==imputed_coords.loc[row1.index[j],"col"]:
#                     neighbor_matrix.loc[row0.index[i],row1.index[j]]=1
#                     neighbor_matrix.loc[row1.index[j],row0.index[i]]=1
#                     flag+=1
#                 if flag>=3:
#                     continue
#
#     # Get not-in-tissue coordinates.
#     neighbor_matrix=neighbor_matrix.loc[:,[str(val[0])+"x"+str(val[1]) for val in integral_coords]]
#     not_in_tissue_coords = []
#     for i in range(len(imputed_coords)):
#         if imputed_coords.index[i] in neighbor_matrix.columns:
#             continue
#         i_row=neighbor_matrix.iloc[i]
#         if sum(i_row!=0)<2:
#             not_in_tissue_coords.append(list(imputed_coords.iloc[i]))
#         if sum(i_row!=0)==2 and sum(i_row==-1)==2:
#             not_in_tissue_coords.append(list(imputed_coords.iloc[i]))
#
#     position_info = [imputed_x, imputed_y, not_in_tissue_coords]
#
#     return position_info

def get_ST_position_info(integral_coords):
    """
    Get information about imputed coordinates for judging whether an imputed spot is in tissue and preserved in the final result. The function is applicable to the situation that integral_coords is related to ST data or other spatial transcriptomic data having spots with matrix arrangement. It can be collocated with getSTtestset() or interpolation_ST().

    Parameter:
    integral_coords: a 2-D ndarray or dataframe of spots coordinates. Each row corresponds to a spot with a barcode, the first column corresponds to row coordinates of the spots and the second column corresponds to column coordinates of the spots. Integral_coords describes row and column coordinates of some original spots that are expected to preserve. If an imputed spot points to an original spot or stays between two adjacent original spots from integral_coords, it will be preserved finally. Usually, integral_coords contains the initial spots of test data without quality control for tissue structural integrity. You can add manually some spots coordinates which you want to preserve.

    Return:
    position_info: information about imputed coordinates.
    position_info = [imputed_x, imputed_y, not_in_tissue_coords, imputed_to_integral_mapping]
    imputed_x: a 2-D ndarray. Row coordinates of imputed spots.
    imputed_y: a 2-D ndarray with the same shape as imputed_x. Column coordinates of imputed spots.
    not_in_tissue_coords: a list of two-element tuples. Row and column coordinates of imputed spots outside tissue.
    imputed_to_integral_mapping: a dictionary mapping imputed coordinates to the nearest integral coordinates.
    """
    integral_coords = np.array(integral_coords)
    delta_x = 1
    delta_y = 1

    # Get coordinates of imputed spots.
    imputed_x, imputed_y = np.mgrid[min(integral_coords[:, 0]):max(integral_coords[:, 0]) + delta_x:delta_x / 2,
                           min(integral_coords[:, 1]):max(integral_coords[:, 1]) + delta_y:delta_y / 2]

    # Count the number of adjacent spots from original data.
    integral_coords = integral_coords.astype(np.float32)
    imputed_barcodes = [str(val[0]) + "x" + str(val[1]) for val in
                        np.vstack((imputed_x.reshape(-1), imputed_y.reshape(-1))).T]
    imputed_coords = pd.DataFrame(np.vstack((imputed_x.reshape(-1), imputed_y.reshape(-1))).astype(np.float32).T,
                                  columns=['row', 'col'], index=imputed_barcodes)
    neighbor_matrix = pd.DataFrame(np.zeros((imputed_coords.shape[0], imputed_coords.shape[0]), dtype=np.int32),
                                   columns=imputed_barcodes, index=imputed_barcodes)

    row1 = imputed_coords[imputed_coords["row"] == min(imputed_coords["row"])].sort_values("col")
    for i in range(len(row1) - 1):
        if row1["col"][i + 1] - row1["col"][i] == delta_y / 2:
            neighbor_matrix.loc[row1.index[i], row1.index[i + 1]] = 1
            neighbor_matrix.loc[row1.index[i + 1], row1.index[i]] = 1
    for row in list(np.array(imputed_x).T[0])[:-1]:
        row0 = imputed_coords[imputed_coords["row"] == row].sort_values("col")
        row1 = imputed_coords[imputed_coords["row"] == row + delta_x / 2].sort_values("col")
        for i in range(len(row1) - 1):
            if row1["col"][i + 1] - row1["col"][i] == delta_y / 2:
                neighbor_matrix.loc[row1.index[i], row1.index[i + 1]] = 1
                neighbor_matrix.loc[row1.index[i + 1], row1.index[i]] = 1
        for i in range(len(row0)):
            for j in range(len(row1)):
                flag = 0
                if abs(imputed_coords.loc[row0.index[i], "col"] - imputed_coords.loc[
                    row1.index[j], "col"]) == delta_y / 2:
                    neighbor_matrix.loc[row0.index[i], row1.index[j]] = -1
                    neighbor_matrix.loc[row1.index[j], row0.index[i]] = -1
                    flag += 1
                if imputed_coords.loc[row0.index[i], "col"] == imputed_coords.loc[row1.index[j], "col"]:
                    neighbor_matrix.loc[row0.index[i], row1.index[j]] = 1
                    neighbor_matrix.loc[row1.index[j], row0.index[i]] = 1
                    flag += 1
                if flag >= 3:
                    continue

    # Get not-in-tissue coordinates.
    neighbor_matrix = neighbor_matrix.loc[:, [str(val[0]) + "x" + str(val[1]) for val in integral_coords]]
    not_in_tissue_coords = []
    for i in range(len(imputed_coords)):
        if imputed_coords.index[i] in neighbor_matrix.columns:
            continue
        i_row = neighbor_matrix.iloc[i]
        if sum(i_row != 0) < 2:
            not_in_tissue_coords.append(list(imputed_coords.iloc[i]))
        if sum(i_row != 0) == 2 and sum(i_row == -1) == 2:
            not_in_tissue_coords.append(list(imputed_coords.iloc[i]))

    # Create mapping of imputed to nearest integral coordinates
    imputed_to_integral_mapping = {}
    for i in range(imputed_x.shape[0]):
        for j in range(imputed_x.shape[1]):
            imputed_coord = (imputed_x[i, j], imputed_y[i, j])
            distances = np.linalg.norm(integral_coords - imputed_coord, axis=1)
            nearest_integral_coord = integral_coords[np.argmin(distances)]
            imputed_to_integral_mapping[imputed_coord] = tuple(nearest_integral_coord)

    position_info = [imputed_x, imputed_y, not_in_tissue_coords]

    return position_info, imputed_to_integral_mapping


def interpolation_ST(counts, coords, method = 'cubic'):
    """    
    A function that apply interpolation method, scipy.interpolate.griddata(), to ST data or other spatial transcriptomic data having spots with matrix arrangement and get interpolated gene-wise expression.
    """     
    counts = np.array(counts)
    coords = np.array(coords)   
    delta_x = 1
    delta_y = 1    
    interpolated_x, interpolated_y = np.mgrid[min(coords[:,0]):max(coords[:,0])+delta_x:delta_x/2, 
                                min(coords[:,1]):max(coords[:,1])+delta_y:delta_y/2]
            
    interpolated_set = [None] * counts.shape[1]
    for i in range(counts.shape[1]):
        interpolated_data = griddata(coords, counts[:,i], (interpolated_x, interpolated_y), method=method,fill_value=0)
        interpolated_set[i] = interpolated_data
    
    interpolated_set = np.array(interpolated_set)
    if np.sum(counts<0)==0:
        interpolated_set[interpolated_set<0] = 0    
    
    return interpolated_set


def img2expr(imputed_img, gene_ids, integral_coords, position_info):
    [imputed_x, imputed_y, not_in_tissue_coords] = position_info

    if type(not_in_tissue_coords)==np.ndarray:
        not_in_tissue_coords=[list(val) for val in not_in_tissue_coords]
        
    integral_barcodes = integral_coords.index
    imputed_counts = pd.DataFrame(np.zeros((imputed_img.shape[1]*imputed_img.shape[2]-len(not_in_tissue_coords),
                                            imputed_img.shape[0])),columns=gene_ids)
    imputed_coords = pd.DataFrame(np.zeros((imputed_img.shape[1]*imputed_img.shape[2]-len(not_in_tissue_coords),
                                            2)),columns=['array_row','array_col'])
    imputed_barcodes = [None] * len(imputed_counts)
    integral_coords = [list(i.astype(np.float32)) for i in np.array(integral_coords)]
    
    flag=0
    for i in range(imputed_img.shape[1]):
        for j in range(imputed_img.shape[2]): 
            
            spot_coords = [imputed_x[i,j],imputed_y[i,j]]
            if spot_coords in not_in_tissue_coords:
                continue
                
            # barcodes
            if spot_coords in integral_coords:
                imputed_barcodes[flag] = integral_barcodes[integral_coords.index(spot_coords)]
            else:
                if int(imputed_x[i,j])==imputed_x[i,j]:
                    x_id = str(int(imputed_x[i,j]))
                else:
                    x_id = str(imputed_x[i,j])
                if int(imputed_y[i,j])==imputed_y[i,j]:
                    y_id = str(int(imputed_y[i,j]))
                else:
                    y_id = str(imputed_y[i,j])
                imputed_barcodes[flag] = x_id+"x"+y_id            
            # counts
            imputed_counts.iloc[flag,:] = imputed_img[:,i,j]
            # coords
            imputed_coords.iloc[flag,:] = spot_coords            
            flag=flag+1
            
    imputed_counts.index=imputed_barcodes
    imputed_coords.index=imputed_barcodes
    
    return imputed_counts, imputed_coords

def enhanceST(train_set, val_set, test_set, epoch=200, batch_size=128, learning_rate=0.001, conf=configs.Config(), gpu=None):
    # Setup configuration and results directory
    # conf = configs.Config()
    conf.epoch = epoch
    conf.batch_size = batch_size
    conf.learning_rate = learning_rate
    conf.test_positive = (np.sum(test_set<0)==0)

    # Run enhanceST
    net = network.Net(train_set, val_set, test_set, conf)
    imputed_img, losses_list, val_losses_list = net.run()

    # epochs = [x[0] for x in net.epoch_loss_list]
    # losses = [x[1] for x in net.epoch_loss_list]
    #
    # plt.figure(figsize=(10, 5))
    # plt.plot(epochs, losses, marker='o', linestyle='-', color='b')
    # plt.title('Training Loss Per Epoch')
    # plt.xlabel('Epoch')
    # plt.ylabel('Average Loss')
    # plt.grid(True)
    # plt.show()
    
    return imputed_img, losses_list, val_losses_list