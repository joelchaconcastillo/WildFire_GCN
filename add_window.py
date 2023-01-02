import os
import numpy as np
import pandas as pd
#import networkx as nx
import zigzag.zigzagtools as zzt
from scipy.spatial.distance import squareform
import scipy.sparse as sp
import dionysus as d
import matplotlib.pyplot as plt
import time
from ripser import ripser
from persim import plot_diagrams, PersImage
path = os.getcwd()

#X: T, N, F
def zigzag_persistence_diagrams(x, alpha, NVertices, scaleParameter, maxDimHoles, sizeWindow):
    GraphsNetX = []
    for t in range(sizeWindow): 
      graphL2 = np.sum((x[t,np.newaxis, :, :]-x[t,:,np.newaxis,:])**2, axis=-1) #compute matrix distance
      graphL2[graphL2==0] = 1e-5  ##weakly connected
      tmp_max = np.max(graphL2)
      graphL2 /= tmp_max  ##normalize  matrix
      graphL2[graphL2>alpha]=0 ##cut off 
      GraphsNetX.append(graphL2)

    # Building unions and computing distance matrices
    print("Building unions and computing distance matrices...")  # Beginning
    MDisGUnions = []
    for i in range(0, sizeWindow - 1):
        # --- To concatenate graphs
        unionAux = []
        MDisAux = np.zeros((2 * NVertices, 2 * NVertices))
        A = GraphsNetX[i] #nx.adjacency_matrix(GraphsNetX[i]).todense()
        B = GraphsNetX[i+1] #nx.adjacency_matrix(GraphsNetX[i + 1]).todense()
        # ----- Version Original (2)
        C = (A + B) / 2
        A[A == 0] = 1.1
        A[range(NVertices), range(NVertices)] = 0
        B[B == 0] = 1.1
        B[range(NVertices), range(NVertices)] = 0
        MDisAux[0:NVertices, 0:NVertices] = A
        C[C == 0] = 1.1
        C[range(NVertices), range(NVertices)] = 0
        MDisAux[NVertices:(2 * NVertices), NVertices:(2 * NVertices)] = B
        MDisAux[0:NVertices, NVertices:(2 * NVertices)] = C
        MDisAux[NVertices:(2 * NVertices), 0:NVertices] = C.transpose()
        # Distance in condensed form
        pDisAux = squareform(MDisAux)
        # --- To save unions and distances
        MDisGUnions.append(pDisAux)  # To save distance matrix
    print("  --- End unions...")  # Ending

    # To perform Ripser computations
    print("Computing Vietoris-Rips complexes...")  # Beginning

    GVRips = []
    for jj in range(0, sizeWindow - 1):
        print(jj)
        ripsAux = d.fill_rips(MDisGUnions[jj], maxDimHoles, scaleParameter)
        GVRips.append(ripsAux)
    print("  --- End Vietoris-Rips computation")  # Ending

    # Shifting filtrations...
    print("Shifting filtrations...")  # Beginning
    GVRips_shift = []
    GVRips_shift.append(GVRips[0])  # Shift 0... original rips01
    for kk in range(1, sizeWindow - 1):
        shiftAux = zzt.shift_filtration(GVRips[kk], NVertices * kk)
        GVRips_shift.append(shiftAux)
    print("  --- End shifting...")  # Ending

    # To Combine complexes
    print("Combining complexes...")  # Beginning
    completeGVRips = zzt.complex_union(GVRips[0], GVRips_shift[1])
    for uu in range(2, sizeWindow - 1):
        completeGVRips = zzt.complex_union(completeGVRips, GVRips_shift[uu])
    print("  --- End combining")  # Ending

    # To compute the time intervals of simplices
    print("Determining time intervals...")  # Beginning
    time_intervals = zzt.build_zigzag_times(completeGVRips, NVertices, sizeWindow)
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
    print("TIME: " + str((time.time() - start_time)) + " Seg ---  " + str(
        (time.time() - start_time) / 60) + " Min ---  " + str((time.time() - start_time) / (60 * 60)) + " Hr ")

    return window_ZPD

# Zigzag persistence image
def zigzag_persistence_images(dgms, resolution = [50,50], return_raw = False, normalization = True, bandwidth = 1., power = 1., dimensional = 0):
    PXs, PYs = np.vstack([dgm[:, 0:1] for dgm in dgms]), np.vstack([dgm[:, 1:2] for dgm in dgms])
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

    return norm_output





maxDimHoles = 2
window = 10
alpha = 0.3
scaleParameter =  0.4
NVertices = 625
data = np.random.rand(window, NVertices, 25)
### for each sample....
zigzag_PD = zigzag_persistence_diagrams(x = data, alpha=alpha, NVertices=NVertices, scaleParameter=scaleParameter, maxDimHoles=maxDimHoles, sizeWindow=window)
zigzag_PI_H0 = zigzag_persistence_images(zigzag_PD, dimensional = 0)
zigzag_PI_H1 = zigzag_persistence_images(zigzag_PD, dimensional = 1)
X_H0.append(zigzag_PI_H0)
X_H1.append(zigzag_PI_H1)

