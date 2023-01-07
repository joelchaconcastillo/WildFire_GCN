import dionysus as d
import numpy as np
import time
from scipy.spatial.distance import squareform
import zigzag.zigzagtools as zzt

class zigzagTDA:
   def __init__(self, NVertices, scaleParameter, maxDimHoles, sizeWindow):
      self.NVertices = NVertices
      self.scaleParameter = scaleParameter
      self.maxDimHoles = maxDimHoles
      self.sizeWindow = sizeWindow

   #X: T, N, F
   def zigzag_persistence_diagrams(self, x, prefix_path):
       GraphsNetX = []
       for t in range(self.sizeWindow): 
         graphL2 = np.sqrt(np.sum((x[t,np.newaxis, :, :]-x[t,:,np.newaxis,:])**2, axis=-1)) #compute matrix distance
         graphL2[graphL2==0] = 1e-5  ## avoid removing zero weights which are good
         graphL2[graphL2>self.scaleParameter]=0
         tmp_max = np.max(graphL2)
         graphL2 /= tmp_max  ##normalize  matrix, this would be better normalizing by rows...
         GraphsNetX.append(graphL2)
       start_time = time.time()
      # Building unions and computing distance matrices
       #print("Building unions and computing distance matrices...")  # Beginning
       MDisGUnions = []
       for i in range(0, self.sizeWindow - 1):
           # --- To concatenate graphs
           MDisAux = np.zeros((2 * self.NVertices, 2 * self.NVertices))
           A = GraphsNetX[i] #nx.adjacency_matrix(GraphsNetX[i]).todense()
           B = GraphsNetX[i+1] #nx.adjacency_matrix(GraphsNetX[i + 1]).todense()
           # ----- Version Original (2)
           C = (A + B) / 2
           A[A == 0] = 10.1
           A[range(self.NVertices), range(self.NVertices)] = 0
           B[B == 0] = 10.1
           B[range(self.NVertices), range(self.NVertices)] = 0
           MDisAux[0:self.NVertices, 0:self.NVertices] = A
           C[C == 0] = 10.1
           C[range(self.NVertices), range(self.NVertices)] = 0
           MDisAux[self.NVertices:(2 * self.NVertices), self.NVertices:(2 * self.NVertices)] = B
           MDisAux[0:self.NVertices, self.NVertices:(2 * self.NVertices)] = C
           MDisAux[self.NVertices:(2 * self.NVertices), 0:self.NVertices] = C.transpose()
           # Distance in condensed form
           pDisAux = squareform(MDisAux)
           # --- To save unions and distances
           MDisGUnions.append(pDisAux)  # To save distance matrix
       #print("  --- End unions...")  # Ending
       
       # To perform Ripser computations
       #print("Computing Vietoris-Rips complexes...")  # Beginning
   
       GVRips = []
       for jj in range(self.sizeWindow - 1):
       #    print(jj)
           ripsAux = d.fill_rips(MDisGUnions[jj], self.maxDimHoles, self.scaleParameter)
           GVRips.append(ripsAux)
       #print("  --- End Vietoris-Rips computation")  # Ending
   
       # Shifting filtrations...
       #print("Shifting filtrations...")  # Beginning
       GVRips_shift = []
       GVRips_shift.append(GVRips[0])  # Shift 0... original rips01
       for kk in range(1, self.sizeWindow - 1):
           shiftAux = zzt.shift_filtration(GVRips[kk], self.NVertices * kk)
           GVRips_shift.append(shiftAux)
       #print("  --- End shifting...")  # Ending
   
       # To Combine complexes
       #print("Combining complexes...")  # Beginning
       completeGVRips = zzt.complex_union(GVRips[0], GVRips_shift[1])
       for uu in range(2, self.sizeWindow - 1):
           completeGVRips = zzt.complex_union(completeGVRips, GVRips_shift[uu])
       #print("  --- End combining")  # Ending
   
       # To compute the time intervals of simplices
       #print("Determining time intervals...")  # Beginning
       time_intervals = zzt.build_zigzag_times(completeGVRips, self.NVertices, self.sizeWindow)
       #print("  --- End time")  # Beginning
   
       # To compute Zigzag persistence
       #print("Computing Zigzag homology...")  # Beginning
       G_zz, G_dgms, G_cells = d.zigzag_homology_persistence(completeGVRips, time_intervals)
       #print("  --- End Zigzag")  # Beginning
   
       # To show persistence intervals
       window_ZPD = []
       # Personalized plot
       for vv, dgm in enumerate(G_dgms):
           #print("Dimension:", vv)
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
       #print("TIME: " + str((time.time() - start_time)) + " Seg ---  " + str((time.time() - start_time) / 60) + " Min ---  " + str((time.time() - start_time) / (60 * 60)) + " Hr ")
   
       return window_ZPD
   
   # Zigzag persistence image
   def zigzag_persistence_images(self, dgms, resolution = [50,50], return_raw = False, normalization = True, bandwidth = 1., power = 1., dimensional = 0):
       if len(dgms) <= dimensional: #validation
           return np.zeros(resolution)
       #print("dimension....", dimensional)
       PXs, PYs = np.vstack([dgm[:, 0:1] for dgm in dgms]), np.vstack([dgm[:, 1:2] for dgm in dgms])
       #print("number selectors..", PXs.shape)
       #print(len(dgms))
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

