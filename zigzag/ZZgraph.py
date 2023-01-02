import dionysus as d
import numpy as np
#import zigzagtools as zzt
import zigzag.zigzagtools as zzt
from math import pi, cos, sin
from random import random
from random import choice
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import os

path = os.getcwd()



def plotting(x, NVertices, alpha, scaleParameter, maxDimHoles, sizeWindow):
   #%% Parameters 
   nameFolderNet = path + 'File'

   #%% Open all sets (point-cloud/Graphs)
   print("Loading data...") # Beginning
#   Graphs = []
#   for i in range(0,sizeWindow):
#      graphL2 = np.sum((x[i,np.newaxis, :, :]-x[i,:,np.newaxis,:])**2, axis=-1) #compute matrix distance
#      graphL2[graphL2==0] = 1e-5  ##weakly connected
#      tmp_max = np.max(graphL2)
#      graphL2 /= tmp_max  ##normalize  matrix
#      graphL2[graphL2>alpha]=0 ##cut off 
#      edgesList = np.array([])
#      for ii in range(100):
#          for jj in range(100):
#              edgesList = np.append(edgesList, [ii, jj, graphL2[ii][jj]])
#       #edgesList = np.loadtxt(nameFolderNet+str(i+1)+".txt") # Load data
##       edgesList = np.loadtxt(nameFolderNet+str(i)+".csv", delimiter=',') # Load data
#      print(edgesList.shape)
#      Graphs.append(edgesList)
#   print("  --- End Loading...") # Ending
   
   #%% Plot Graph
   GraphsNetX = []
   plt.figure(num=None, figsize=(16, 1.5), dpi=80, facecolor='w', edgecolor='k')
   for i in range(0,sizeWindow):
       print(i)
       g = nx.Graph()
       graphL2 = np.sum((x[i,np.newaxis, :, :]-x[i,:,np.newaxis,:])**2, axis=-1) #compute matrix distance
#       graphL2[graphL2==0] = 1e-5  ##weakly connected
       tmp_max = np.max(graphL2)
       graphL2 /= tmp_max  ##normalize  matrix
       graphL2[graphL2>alpha]=0 ##cut off 
       graphL2[range(NVertices), range(NVertices)]=0 ##no self-cycles..
       g = nx.from_numpy_matrix(graphL2)
       GraphsNetX.append(g)
       plt.subplot(1, sizeWindow, i+1)
       plt.title(str(i))
       pos = nx.circular_layout(GraphsNetX[i])
       nx.draw(GraphsNetX[i], pos, node_size=15, edge_color='r') 
       #nx.draw_circular(GraphsNetX[i], node_size=15, edge_color='r') 
       labels = nx.get_edge_attributes(GraphsNetX[i], 'weight')
       for lab in labels:
           labels[lab] = round(labels[lab],2)
       nx.draw_networkx_edge_labels(GraphsNetX[i], pos, edge_labels=labels, font_size=5)
#       plt.show()
#       exit(0)
   
   plt.savefig('Graphs.pdf', bbox_inches='tight')
#   exit(0)   
   #%% Building unions and computing distance matrices 
   print("Building unions and computing distance matrices...") # Beginning
   GUnions = []
   MDisGUnions = []
   for i in range(0,sizeWindow-1):
       # --- To concatenate graphs
       unionAux = []
       MDisAux = np.zeros((2*NVertices, 2*NVertices))
       A = nx.adjacency_matrix(GraphsNetX[i]).todense()
       B = nx.adjacency_matrix(GraphsNetX[i+1]).todense()
       
       # ----- Version Original (2)
       C = (A+B)/2
       A[A==0] = 1.0
       A[range(NVertices), range(NVertices)] = 0
       B[B==0] = 1.0
       B[range(NVertices), range(NVertices)] = 0
       MDisAux[0:NVertices,0:NVertices] = A
       C[C==0] = 1.0
       C[range(NVertices), range(NVertices)] = 0
       MDisAux[NVertices:(2*NVertices),NVertices:(2*NVertices)] = B
       MDisAux[0:NVertices,NVertices:(2*NVertices)] = C
       MDisAux[NVertices:(2*NVertices),0:NVertices] = C.transpose()
       
       # Distance in condensed form 
       pDisAux = squareform(MDisAux)
       
       # --- To save unions and distances
       GUnions.append(unionAux) # To save union
       MDisGUnions.append(pDisAux) # To save distance matrix
   print("  --- End unions...") # Ending
   
   #%% To perform Ripser computations
   print("Computing Vietoris-Rips complexes...") # Beginning
   GVRips = []
   for i in range(0,sizeWindow-1):
       ripsAux = d.fill_rips(MDisGUnions[i], maxDimHoles, scaleParameter) 
       GVRips.append(ripsAux)
   print("  --- End Vietoris-Rips computation") # Ending
   
   #%% Shifting filtrations...
   print("Shifting filtrations...") #Beginning
   GVRips_shift = []
   GVRips_shift.append(GVRips[0]) # Shift 0... original rips01 
   for i in range(1,sizeWindow-1):
       shiftAux = zzt.shift_filtration(GVRips[i], NVertices*i)
       GVRips_shift.append(shiftAux)
   print("  --- End shifting...") # Ending
   
   #%% To Combine complexes
   print("Combining complexes...") # Beginning
   completeGVRips = zzt.complex_union(GVRips[0], GVRips_shift[1]) 
   for i in range(2,sizeWindow-1):
       completeGVRips = zzt.complex_union(completeGVRips, GVRips_shift[i]) 
   print("  --- End combining") # Ending
   
   #%% To compute the time intervals of simplices
   print("Determining time intervals...") # Beginning
   time_intervals = zzt.build_zigzag_times(completeGVRips, NVertices, sizeWindow)
   print("  --- End time") # Beginning
   
   #%% To compute Zigzag persistence
   print("Computing Zigzag homology...") # Beginning
   G_zz, G_dgms, G_cells = d.zigzag_homology_persistence(completeGVRips, time_intervals)
   print("  --- End Zigzag") # Beginning
   
   #%% To show persistence intervals
   print("Persistence intervals:")
   print("++++++++++++++++++++++")
   print(G_dgms)
   for i, dgm in enumerate(G_dgms):
       print(i)
       for p in dgm:
           print(p)
   print("++++++++++++++++++++++")
   for i,dgm in enumerate(G_dgms):
       print("Dimension:", i)
       if(i<2):
           for p in dgm:
               print(p)
   for i,dgm in enumerate(G_dgms):
       print("Dimension:", i)
       if(i<2):
           d.plot.plot_bars(G_dgms[i],show=True)
   
   # %% Personalized plot
   for i,dgm in enumerate(G_dgms):
       print("Dimension:", i)
       if(i<2):
           matBarcode = np.zeros((len(dgm), 2)) 
           k = 0
           for p in dgm:
               #print("( "+str(p.birth)+"  "+str(p.death)+" )") 
               matBarcode[k,0] = p.birth
               matBarcode[k,1] = p.death
               k = k + 1
           matBarcode = matBarcode/2   ## final PD ##
           print(matBarcode)
           for j in range(0,matBarcode.shape[0]):
               plt.plot(matBarcode[j], [j,j], 'b')
           #my_xticks = [0,1,2,3,4,5,6,7,8,9,10,11]
           #plt.xticks(x, my_xticks)
           plt.xticks(np.arange(10))
           plt.grid(axis='x', linestyle='-')
           plt.savefig('BoxPlot'+str(i)+'.pdf', bbox_inches='tight')
           plt.show() 
           
   #%%
   
   #%%
   for s in GVRips[0]:
       print(s)
   # %%
