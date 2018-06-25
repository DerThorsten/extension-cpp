from torch.utils.data.dataset import Dataset
from torchvision import transforms

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import inferno

import torch.nn as nn
from inferno.io.box.cifar import get_cifar10_loaders
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from inferno.extensions.layers.convolutional import ConvELU2D
from inferno.extensions.layers.reshape import Flatten

import os
import vigra
import numpy
import nifty.cgp
import nifty.segmentation
import nifty.ground_truth

from inferno.extensions.layers.convolutional import *


from unet import *

from cpp.spacc import *


class LossWrapper(nn.Module):
    def __init__(self):
        super(LossWrapper, self).__init__()
        #self.loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.5, 2.0]), reduce=False)
        self.cell1_loss = torch.nn.BCELoss(reduce=False)
        self.lifted_loss = torch.nn.BCELoss(reduce=False)
     

        self.cell0_3_loss = torch.nn.CrossEntropyLoss(reduce=False)
        if False:
            self.cell0_4_loss = torch.nn.CrossEntropyLoss(reduce=False)

    def forward(self,thepred, all_gt):
        cell_1_prediction, j3pred, lifted_pred = thepred
        #cell_1_prediction, j3pred = thepred
        cell_1_gt, cell_1_lw, cell_0_3_gt, cell_0_3_lw, lifted_edge_gt, lifted_edge_loss_weight = all_gt
        
        cell_1_prediction = torch.squeeze(cell_1_prediction)
        lifted_pred = torch.squeeze(lifted_pred)
        lifted_edge_gt = torch.squeeze(lifted_edge_gt)
        cell_1_gt = torch.squeeze(cell_1_gt)
        cell_1_lw = torch.squeeze(cell_1_lw)
        cell_0_3_gt = torch.squeeze(cell_0_3_gt)
        #cell_0_4_gt = torch.squeeze(cell_0_4_gt)
        cell_0_3_lw = torch.squeeze(cell_0_3_lw)
        lifted_edge_loss_weight = torch.squeeze(lifted_edge_loss_weight)
        #cell_0_4_lw = torch.squeeze(cell_0_4_lw)

        ############################
        # CELL 1 / BOUNDARY LOSS
        ############################
        lc1 =  self.cell1_loss(cell_1_prediction, cell_1_gt)
        lc1 = torch.sum(lc1 * cell_1_lw)/cell_1_prediction.size(0)


        ############################
        # LIFTED EDGE LOSS
        ############################
        #tmp = lifted_pred.cpu().detach().numpy()
        #gt = lifted_edge_gt.cpu().detach().numpy()
        #print("lifted_pred",tmp.min(),tmp.max())
        #print("gt",gt.min(),gt.max())
        if True:
            ll =  self.lifted_loss(lifted_pred, lifted_edge_gt)
            ll = torch.sum(ll * lifted_edge_loss_weight)/lifted_pred.size(0)



        ############################
        # CELL0 3/ J3 LOSS
        ############################
    
        lc03 =  self.cell0_3_loss(j3pred, cell_0_3_gt)
        lc03 = torch.sum(lc03 * cell_0_3_lw)/cell_0_3_lw.size(0)

        ############################
        # CELL0 4/ J4 LOSS
        ############################
        #print("j4pred",j4pred.size(), "lw",cell_0_4_lw.size())
        #lc04 =  self.cell0_4_loss(j4pred, cell_0_4_gt)
        #lc04 = torch.sum(lc04 * cell_0_4_lw)/cell_0_4_lw.size(0)

        ############################
        # Total LOSS
        ############################
        total_loss = lc1 + 0.01*lc03 + 0.01*ll
        #print("THELOSS: ",float(total_loss.item()))
        #print("     c1",float(lc1.item()), "c0_3",float(lc03.item()) , "ll",float(ll.item()))

        return total_loss

class MyNN(nn.Module):
    def __init__(self, in_channels, out_channels=None, hidden_gain=2, activated=True,p=0.2):
        super(MyNN, self).__init__()

        self.activated = activated
        self.in_channels = in_channels
        self.out_channels = out_channels
        if out_channels is None:
            self.out_channels = in_channels
        self.hidden_gain = hidden_gain

        
        self.hidden_1 = nn.Linear(in_channels,     in_channels * hidden_gain)
        #self.dropout  = nn.Dropout(p=p)
        self.hidden_2 = nn.Linear(in_channels * hidden_gain, self.out_channels)
    
        self.nl = nn.ELU()

    def forward(self, input):
        assert input.size(1) == self.in_channels
        out = self.hidden_1(input)
        out = self.nl(out)
        #out = self.dropout(out)
        out = self.hidden_2(out)
        if self.activated:
            out  = self.nl(out)
        assert out.size(1) == self.out_channels
        return out

class EdgeFeatNodeFeatModule(nn.Module):
    def __init__(self, in_channels_e, in_channels_uv):
        super(EdgeFeatNodeFeatModule, self).__init__()

        n_in = in_channels_e + 2*in_channels_uv
        n_hidden = n_in * 2 

        # self.n_jpool_in = n_hidden
        #self.n_jpool_out  = 3 * self*n_jpool_in
        #self.n_jpool_bn   = 5

        self.in_channels_e = in_channels_e
        self.in_channels_uv = in_channels_uv
        self.out_channels = n_hidden 
        self.hidden_1 = nn.Linear(n_in,         n_hidden)
        self.hidden_2 = nn.Linear(n_hidden,     n_hidden)
        self.nl       = nn.ELU()

        #self.j_pooled_bn = nn.Linear(n_in,     n_hidden)

        #self.j_pool = JunctionPoolModule()
        #self.n_in = n_in
    def forward(self, f_e, f_eu, f_ev, cell_0_bounds):

        assert f_e.size(1)  == self.in_channels_e
        assert f_eu.size(1) == self.in_channels_uv
        assert f_ev.size(1) == self.in_channels_uv

        a = torch.cat([f_e, f_eu, f_ev], 1)
        b = torch.cat([f_e, f_ev, f_eu], 1)
        a = self.hidden_1(a)
        b = self.hidden_1(b)
        a = self.nl(a)
        b = self.nl(b)

        a = self.hidden_2(a)
        b = self.hidden_2(b)
        a = self.nl(a)
        b = self.nl(b)

        res = a + b
        #res = self.nl(res)
        assert res.size(1) == self.out_channels
        return res



class LiftedEdgeModule(nn.Module):
    def __init__(self, in_channels_uv):
        super(LiftedEdgeModule, self).__init__()

        n_in = 2*in_channels_uv
        n_hidden = n_in #* 2 

        # self.n_jpool_in = n_hidden
        #self.n_jpool_out  = 3 * self*n_jpool_in
        #self.n_jpool_bn   = 5

        self.in_channels_uv = in_channels_uv
        self.out_channels = n_hidden 
        self.hidden_1 = nn.Linear(n_in,         n_hidden)
        self.hidden_2 = nn.Linear(n_hidden,     n_hidden)
        self.nl       = nn.ELU()

        #self.j_pooled_bn = nn.Linear(n_in,     n_hidden)

        #self.j_pool = JunctionPoolModule()
        #self.n_in = n_in
    def forward(self, f_eu, f_ev):
        #print("insize",f_eu.size(1),"should ",self.in_channels_uv)
        #print("insize",f_ev.size(1),"should ",self.in_channels_uv)
        assert f_eu.size(1) == self.in_channels_uv
        assert f_ev.size(1) == self.in_channels_uv

        a = torch.cat([f_eu, f_ev], 1)
        b = torch.cat([f_ev, f_eu], 1)
        a = self.hidden_1(a)
        b = self.hidden_1(b)
        a = self.nl(a)
        b = self.nl(b)
        a = self.hidden_2(a)
        b = self.hidden_2(b)
        a = self.nl(a)
        b = self.nl(b)

        res = a + b
        #res = self.nl(res)
        assert res.size(1) == self.out_channels
        return res





class ExtractFromJ(nn.Module):
    def __init__(self, jsize, in_channels):
        super(ExtractFromJ, self).__init__()
        assert jsize in [3,4]
        self.jsize = jsize
        self.in_channels = in_channels

        if (in_channels//jsize)*jsize  != in_channels:
            raise RuntimeError("in_channels (%d) is not dividable by %d"%(in_channels, jsize))

        self.per_cell1_channels = self.in_channels // self.jsize
        self.out_channels = 2 * self.per_cell1_channels
    def forward(self, features, cell1Masks):

        if features.size(1) != self.in_channels:
            raise RuntimeError("expected %d channels, got %d"%(self.in_channels,features.size(1)))
       

        nj = cell1Masks.size(0)  
        w = cell1Masks.size(2)  
        h = cell1Masks.size(3)  



        flatCell1Masks = cell1Masks.view(nj,self.jsize,-1)
   
        #cellSizes |J| x jsize 
        cellSizes =  torch.sum(flatCell1Masks, 2)

    
        features = features.view(nj, self.per_cell1_channels, self.jsize, -1)

        # broadcasting multiplication

        broadcastable_flatCell1Mask = flatCell1Masks.float().view(nj, 1, self.jsize, -1)

      
        # masked_unet_output  |J| x per_cell1_channels x jsize x W*H
        masked_features = features * broadcastable_flatCell1Mask


        # extract features
        # cell_1_features_sum  |J| x per_cell1_channels x jsize
        cell_1_features_sum = torch.sum(masked_features, 3)


        # cell_1_features_mean  |J| x per_cell1_channels x jsize
        cell_1_features_mean = cell_1_features_sum / cellSizes.float().view(nj, 1, self.jsize)

        # out  |J| x 2*per_cell1_channels x jsize
        out = torch.cat([cell_1_features_sum, cell_1_features_mean], 1)

        return out




class BatchJunctionModule(nn.Module):
    def __init__(self, jsize, per_cell1_channels=4, depth=3, unet_depth=3, gain=2, p=0.15):


        super(BatchJunctionModule, self).__init__()
        assert jsize in [3,4]
        self.jsize = jsize   

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        #self.dropout = nn.Dropout2d(p=p)
        conv_blocks = []
        extract_blocks = []
        in_channels = 5+self.jsize
        self.depth = depth

        self.n_cell_0_feat_out = 0

        for i in range(depth):

            if i == 0 :
                out_channels = in_channels * self.jsize
            else:
                out_channels = in_channels * gain

            #conv_block = DenseBlock(in_channels=in_channels, out_channels=out_channels, size=4, out_per_conv=3)
            conv_block = DenseBlock(in_channels=in_channels, out_channels=out_channels)
            in_channels = out_channels


            conv_blocks.append(conv_block)
            # extractor blocks
            extractor = ExtractFromJ(in_channels=out_channels, jsize=self.jsize)
            extract_blocks.append(extractor)

            self.n_cell_0_feat_out += (out_channels//self.jsize)*2
            image_data_n_channels = out_channels

            #in_channels *= gain
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.extract_blocks = nn.ModuleList(extract_blocks)
        
        # input size of fully
        n_in_fully  = 4*4*image_data_n_channels
        n_out_fully = self.jsize * 20

        self.n_cell_0_feat_out += 20

        self.fully = MyNN(in_channels=n_in_fully, out_channels=n_out_fully)



    def forward(self, masks, image_data):

        # masks       |J| x JSIZE + 1 x W x H
        # image_data  |J| x  c        x W x H
        nj = masks.size(0)  
        w = masks.size(2)  
        h = masks.size(3)  
        # mask magic
        cell1Masks = masks[:, 0 : self.jsize, : , :].float()

        #flatCell1Masks |J| x jsize x w*h
    
        flatCell1Masks = cell1Masks.contiguous().view(nj,self.jsize,-1)
   
        #cellSizes |J| x jsize 
        cellSizes =  torch.sum(flatCell1Masks, 2)




        current_input = torch.cat([image_data,masks.float()],1)
        current_masks = cell1Masks

        
        # collect the extracted jfeat
        jfeat = []

        for i in range(self.depth):

            # run conv block

            
            

            conv_res = self.conv_blocks[i](current_input)
            #conv_res = self.dropout(conv_res)

            # extract features
            eres = self.extract_blocks[i](conv_res, current_masks)
            #print("eres", eres.size())
            jfeat.append(eres)

            # pool down
            if i + 1 != self.depth:
                current_input = self.max_pool(conv_res)
                current_masks = self.max_pool(current_masks)
            else:
                current_input = conv_res

        # from here we do smth like a fully connected layer?
        
        input_for_fully = current_input.view(nj, -1)
        fully_res = self.fully(input_for_fully)
        fully_res = fully_res.view(nj, -1, self.jsize)

        jfeat.append(fully_res)
        jfeat = torch.cat(jfeat,1)

        #print("jfeat",jfeat.size())
        assert jfeat.size(1) == self.n_cell_0_feat_out
        return jfeat

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # self.layer1 = BNReLUConv2D(kernel_size=3, in_channels=4,  out_channels=10)
        # self.layer2 = BNReLUConv2D(kernel_size=3, in_channels=10, out_channels=10)
        # self.layer3 = ConvELU2D(kernel_size=3,    in_channels=10, out_channels=10)
        
        self.unet    = ResBlockUnet(in_channels=5)

        #self.dropout = nn.Dropout2d(p=0.2)

        self.acc = Cell12AccModule(in_channels=self.unet.out_channels)

        self.nn_cell_1  = MyNN(in_channels=self.acc.out_channels)
        self.nn_cell_2  = MyNN(in_channels=self.acc.out_channels)

      
        self.edge_hidden = EdgeFeatNodeFeatModule(self.acc.out_channels, self.acc.out_channels)
        self.lifted_edges_op = LiftedEdgeModule(self.acc.out_channels)

        self.lifted_edge_pred = MyNN(in_channels=self.lifted_edges_op.out_channels,
                              out_channels=1, activated=False)

        self.hidden_3    = nn.Linear(self.edge_hidden.out_channels, 1)

        self.sigmoid     = nn.Sigmoid()
        self.softmax     = nn.Softmax(dim=1)

        self.j_pool    = JunctionPoolModule()
        self.j_pool_nn = MyNN(in_channels=2*self.edge_hidden.out_channels,
                              out_channels=self.edge_hidden.out_channels)


        self.j3 = BatchJunctionModule(jsize=3)
        
        self.nn_j3  = MyNN(in_channels=(self.j3.n_cell_0_feat_out + self.edge_hidden.out_channels + 1)*3, out_channels= 5, activated=False)


    def forward(self, padded_image,  padded_cell_masks, 
                cell_0_bounds, cell_1_bounds, 
                cell_1_sizes, cell_2_sizes,
                cell0_3_bounds, c03, fc03,
                lifted_edges):

        lifted_edges = torch.squeeze(lifted_edges)
        cell0_3_bounds = torch.squeeze(cell0_3_bounds)-1



        padded_mask_1 = padded_cell_masks[:,1,...]
        padded_mask_2 = padded_cell_masks[:,2,...]

        assert isinstance(cell_1_sizes,     (torch.cuda.IntTensor, torch.IntTensor))
        assert isinstance(cell_2_sizes,     (torch.cuda.IntTensor, torch.IntTensor))

        is_boundary = torch.clamp(padded_mask_1, min=0, max=1).float() - 0.5

        input = torch.cat([padded_image, torch.unsqueeze(is_boundary, 0)], 1)
        #input = self.dropout(input)

        out = self.unet(input)


        cell_1_features, cell_2_features = self.acc(out, 
            torch.squeeze(padded_mask_1), 
            torch.squeeze(padded_mask_2), 
            torch.squeeze(cell_1_sizes), 
            torch.squeeze(cell_2_sizes)
        )
        
        cell_1_features_new = self.nn_cell_1(cell_1_features) + cell_1_features
        cell_2_features_new = self.nn_cell_2(cell_2_features) + cell_2_features

            



        #cell_1_features_new 
        #cell_2_features_new




        cell_1_bounds = cell_1_bounds.squeeze().long()
        u = cell_1_bounds[:,0] - 1
        v = cell_1_bounds[:,1] - 1



        cell_2_features_u = cell_2_features_new[u, :]
        cell_2_features_v = cell_2_features_new[v, :]

        if True:

            lu = lifted_edges[:,0] - 1
            lv = lifted_edges[:,1] - 1
            cell_2_features_lu = cell_2_features_new[lu, :]
            cell_2_features_lv = cell_2_features_new[lv, :]
            lifted_feat = self.lifted_edges_op(cell_2_features_lu, cell_2_features_lv)
            lifted_pred = self.lifted_edge_pred(lifted_feat)
            lifted_pred = self.sigmoid(lifted_pred)


        out = self.edge_hidden(cell_1_features_new, cell_2_features_u, cell_2_features_v, torch.squeeze(cell_0_bounds))
        
        if False:
            jpool = self.j_pool(out, cell_0_bounds.squeeze())
            jpool = jpool.view(out.size(0),-1)
            cell_1_feat = self.j_pool_nn(jpool) + out
        else:
            cell_1_feat = out


        cell_1_pred = self.hidden_3(cell_1_feat)
        cell_1_pred = self.sigmoid(cell_1_pred)


        # extract the j represenation
        # |C3| x F x 1
        e30 =cell_1_feat[cell0_3_bounds[:,0],:].unsqueeze(2)
        e31 =cell_1_feat[cell0_3_bounds[:,1],:].unsqueeze(2)
        e32 =cell_1_feat[cell0_3_bounds[:,2],:].unsqueeze(2)

        p30 =cell_1_pred[cell0_3_bounds[:,0],:].unsqueeze(2)
        p31 =cell_1_pred[cell0_3_bounds[:,1],:].unsqueeze(2)
        p32 =cell_1_pred[cell0_3_bounds[:,2],:].unsqueeze(2)

        # |C3| x edge_hidden.out x 3 
        j3_e_feat = torch.cat([e30,e31,e32], 2)
        j3_p_feat = torch.cat([p30,p31,p32], 2)






        #j3_feat =  |J| x 2*per_cell1_channels x 3
        j3_feat = self.j3(masks=torch.squeeze(c03), image_data=torch.squeeze(fc03))
        

        j3_feat = torch.cat([j3_feat, j3_e_feat, j3_p_feat], 1)
        #j3_feat =  |J| x (2*per_cell1_channels  +edge_liear.out_channels)* 3
        j3_feat = j3_feat.view(j3_feat.size(0), -1)
        


        j3_pred = self.nn_j3(j3_feat)
        j3_pred = self.softmax(j3_pred)



        return cell_1_pred, j3_pred,lifted_pred
