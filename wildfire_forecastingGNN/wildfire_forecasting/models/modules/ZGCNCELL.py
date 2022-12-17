import torch
import torch.nn as nn
import torch.nn.functional as F

from wildfire_forecasting.models.modules.DNN import CNN

#from wildfire_forecasting.models.modules.ZGCN import TLSGCNCNN, TFLSGCNCNN
# GRU with the output from ZGCN.py
class NLSGCRNCNNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, window_len, link_len, embed_dim):
        super(NLSGCRNCNNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = TFLSGCNCNN(dim_in+self.hidden_dim, 2*dim_out, window_len, link_len, embed_dim)
        self.update = TFLSGCNCNN(dim_in+self.hidden_dim, dim_out, window_len, link_len, embed_dim)

    def forward(self, x, state, x_full, node_embeddings, zigzag_PI):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1) #x + state
        z_r = torch.sigmoid(self.gate(input_and_state, x_full, node_embeddings, zigzag_PI))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, x_full, node_embeddings, zigzag_PI))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)

# ZPI * Spatial GC Layer || ZPI * Feature Transformation (on Temporal Features)
# with less parameters
class TFLSGCNCNN(nn.Module):
    def __init__(self, dim_in, dim_out, window_len, link_len, embed_dim):
        super(TFLSGCNCNN, self).__init__()
        self.link_len = link_len
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, link_len, dim_in, int(dim_out/2)))
        if (dim_in-25)%16 ==0: ##differentiate initial to inner cells
           self.weights_window = nn.Parameter(torch.FloatTensor(embed_dim, 1, int(dim_out / 2)))
        else:
           self.weights_window = nn.Parameter(torch.FloatTensor(embed_dim, int(dim_in/2), int(dim_out / 2)))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.T = nn.Parameter(torch.FloatTensor(window_len))
        self.cnn = CNN(int(dim_out / 2))
    def forward(self, x, x_window, node_embeddings, zigzag_PI):
        #S1: Laplacian construction
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]

        #S2: Laplacianlink
        for k in range(2, self.link_len):
            support_set.append(torch.mm(supports, support_set[k-1]))
        supports = torch.stack(support_set, dim=0)

        #S3: spatial graph convolution
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool) #N, link_len, dim_in, dim_out/2
        bias = torch.matmul(node_embeddings, self.bias_pool) #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x) #B, link_len, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3) #B, N, link_len, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) #B, N, dim_out/2

        #S4: temporal feature transformation
        weights_window = torch.einsum('nd,dio->nio', node_embeddings, self.weights_window)  #N, dim_in, dim_out/2
        x_w = torch.einsum('btni,nio->btno', x_window, weights_window)  #B, T, N, dim_out/2
        x_w = x_w.permute(0, 2, 3, 1)  #B, N, dim_out/2, T
        x_wconv = torch.matmul(x_w, self.T)  #B, N, dim_out/2

        #S5: zigzag persistence representation learning
        topo_cnn = self.cnn(zigzag_PI) #B, dim_out/2, dim_out/2
        x_tgconv = x_gconv #torch.einsum('bno,bo->bno',x_gconv, topo_cnn)
        x_twconv = x_wconv #torch.einsum('bno,bo->bno',x_wconv, topo_cnn)

        #S6: combination operation
        x_gwconv = torch.cat([x_tgconv, x_twconv], dim = -1) + bias #B, N, dim_out
        return x_gwconv
