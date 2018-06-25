from torch.utils.data.dataset import Dataset
from torchvision import transforms

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import nifty
import inferno

import pylab
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
import random
from inferno.extensions.layers.convolutional import *

from collections import OrderedDict
from sp import *
from loss_weight import *

def pad_bsd_to_(img, ts = 2048, mode='reflect', channels_front=False):
    dtype_in = img.dtype
    if channels_front:
        shape = img.shape[1:3]
    else:
        shape = img.shape[0:2]
    diff  = [ ts - s for s in shape]
    ps = [d//2 for d in diff]
    pe = [d-p for d,p in zip(diff, ps)]
    p = [ (s,e) for s,e in zip(ps, pe)]

    if img.ndim == 3:
        if channels_front:
            pimg = numpy.zeros([img.shape[0], ts,ts])
        else:
            pimg = numpy.zeros([ts,ts, img.shape[2]])

        if channels_front:
            for c in range(img.shape[0]):
                pimg[c,...] = numpy.pad(img[c,...], pad_width=p, mode=mode)
        else:
            for c in range(img.shape[2]):
                pimg[...,c] = numpy.pad(img[...,c], pad_width=p, mode=mode)
    elif img.ndim == 2:
        pimg = numpy.pad(img, pad_width=p, mode=mode)
    else:
        assert False
    pimg = numpy.require(pimg, dtype=dtype_in)
    return pimg,ps


from cpp.spacc import *



def readImgC(filename, dtype=None):
    img = vigra.impex.readImage(filename).squeeze()
    if dtype is not None:
        img = numpy.require(img, requirements=['C'], dtype=dtype)
    else:
        img = numpy.require(img, requirements=['C'])
    return img



# def make_sp(self)


def extractLiftedEdges(uvIds, d=3):
     
    max_node = uvIds.max() + 1
    g = nifty.graph.undirectedGraph(max_node + 1)
    g.insertEdges(uvIds.astype('uint64'))

    lifteduv = g.graphNeighbourhood(maxDistance=d, suppressGraphEdges=True)
    #print("lifteduv.max() ",lifteduv.max(),"max_node",max_node)
    assert lifteduv.max() <= max_node
    assert lifteduv.min() >= 1
    return lifteduv



class Bsd500Sp(Dataset):
    def __init__(self,bsd_root, pmap_root,split, joint_transformation=None):
        
        self.bsd500_mean = numpy.array([110.797, 113.003, 93.5948], dtype='float32')
        self.bsd500_std = numpy.array([63.6275, 59.7415, 61.6398], dtype='float32')


        self.pmap_root = pmap_root
        self.bsd_root = bsd_root
        self.split = split

        self.rgb_folder = os.path.join(bsd_root, 'data', 'images', split)
        self.pmap_folder = os.path.join(pmap_root, split)
        self.gt_folder = os.path.join(bsd_root, 'data', 'groundTruth', split)

        # get all images filenames
        self.img_filenames = []
        self.img_numbers = []
        for file in os.listdir(self.rgb_folder):
            if file.endswith('.jpg'):
                filename = os.path.join(self.rgb_folder, file)
                self.img_filenames.append(filename)
                self.img_numbers.append(file[:-4])


        #...
        self.joint_transformation = joint_transformation
    
    def __getitem__(self, index):
        odict = self.getitemimpl(index)
        res = []
        for key in odict.keys():
            res.append(odict[key])
        return tuple(res)


    def make_cell_1_gt(self, tgrid, sp, cell_1_bounds,lifted_edges, gt_stack, cell_1_sizes, cell_2_sizes):
        soft_cell_1_gt = numpy.zeros(tgrid.numberOfCells[1], dtype='float32')
        soft_lifted_edge_gt = numpy.zeros(lifted_edges.shape[0], dtype='float32')

        for i in range(gt_stack.shape[0]):
            labels = gt_stack[i,...]
            labels = numpy.require(labels.view(numpy.ndarray), requirements=['C'])

            overlap = nifty.ground_truth.overlap(segmentation=sp, 
                                           groundTruth=labels)

            soft_cell_1_gt      += overlap.differentOverlaps(cell_1_bounds)
            soft_lifted_edge_gt += overlap.differentOverlaps(lifted_edges)

        soft_cell_1_gt /= gt_stack.shape[0]
        hard_gt = numpy.round(soft_cell_1_gt, 1)
        #semi_hard_gt = 0.01*soft_cell_1_gt + 0.99*hard_gt

        soft_lifted_edge_gt /= gt_stack.shape[0]
        hard_lifted_edge_gt = numpy.round(soft_lifted_edge_gt, 1)
        #emi_hard_lifted_edge_gt = 0.01*soft_lifted_edge_gt + 0.99*hard_lifted_edge_gt

        slu = cell_1_sizes[lifted_edges[:,0]-1]
        slv = cell_1_sizes[lifted_edges[:,1]-1]
        lifted_edge_sizes = numpy.minimum(slu, slv)

        return hard_gt,        get_cell_1_loss_weight(soft_cell_1_gt, sizes=cell_1_sizes), \
               hard_lifted_edge_gt, get_cell_1_loss_weight(soft_lifted_edge_gt, sizes=lifted_edge_sizes)

    def make_cell_masks(self, tgrid):


        ftgrid = nifty.cgp.FilledTopologicalGrid2D(tgrid)


        cell_0_mask = ftgrid.cellMask([True,False,False])
        cell_0_mask[cell_0_mask!=0] -=  ftgrid.cellTypeOffset[0]

        cell_1_mask = ftgrid.cellMask([False,True,False])
        cell_1_mask[cell_1_mask!=0] -=  ftgrid.cellTypeOffset[1]

        # 4d images?
        cell_2_mask = ftgrid.cellMask([False,False,True])
        cell_2_mask[cell_2_mask!=0] -=  ftgrid.cellTypeOffset[2]


        cell_masks = numpy.concatenate([
            cell_0_mask[None,...],
            cell_1_mask[None,...],
            cell_2_mask[None,...]
        ], axis=0)

        return cell_masks


    def getitemimpl(self, index, tt_augment=False, return_test_data=False, sp=None):



        # read rgb and make it big
        img_raw = readImgC(self.img_filenames[index], dtype='float32')
        small_shape = img_raw.shape[0:2]
        img_raw_big = vigra.sampling.resize(vigra.taggedView(img_raw, "xyc"), [2*s -1 for s in small_shape])
        img_raw_big = numpy.rollaxis(img_raw_big,2,0)

        # read  and normalize the pmap
        pmap_filename = os.path.join(self.pmap_folder, self.img_numbers[index] + '.png')
        pmap = 1.0 - readImgC(pmap_filename, dtype='float32')/255.0

        # read gt stack from disc
        gt_filename = os.path.join(self.gt_folder, self.img_numbers[index] + '.mat')
        gt_stack = self._get_bsd_gt_stack(gt_filename)
        gt_stack = numpy.require(gt_stack, requirements=['C'])
        gt_stack = numpy.rollaxis(gt_stack,2,0)# ...]
        # basic transformations
        # - this can flip / mirror the img/pmap/gt
   



        ##############################
        # TRANSFORMS
        ##############################

        #False
        pmap = pmap#[None,...]
        #[...]

    

        # for t in self.joint_transformation.transforms:
        #     try:
        #         t.build_random_variables(imshape=pmap.squeeze().shape)
        #     except TypeError:
        #         t.build_random_variables()
        if self.joint_transformation is not None and sp is None:
            pmap, img_raw_big,  gt_stack = self.joint_transformation(pmap, img_raw_big, gt_stack)
        # img_raw = self.joint_transformation(img_raw)
        # gt_stack = self.joint_transformation(gt_stack)




        if False:
            pylab.imshow(pmap[...])
            pylab.show()

            pylab.imshow(img_raw_big[0,...])
            pylab.show()

            pylab.imshow(gt_stack[1,...])
            pylab.show()




        if sp is not None:
            sp = numpy.require(sp, dtype='uint32')
            sp = vigra.analysis.labelImage(sp)
        else:
            sp = bsd_sp(img_raw_big, pmap, n_sp = 1000, tt_augment=tt_augment,train=self.split=='train')
        sp = numpy.require(sp.view(numpy.ndarray), requirements=['C'])
      
        # assert sp.shape == small_shape
        # assert list(img_raw_big.shape[1:3]) == [2*s-1 for s in small_shape]
        sp = sp.squeeze()
        sp = numpy.require(sp.view(numpy.ndarray), requirements=['C'], dtype='uint64')
        #img_raw_big = numpy.require(img_raw_big.view(numpy.ndarray), requirements=['C'], dtype='float')

    
        tgrid = nifty.cgp.TopologicalGrid2D(sp)
        cell_masks = self.make_cell_masks(tgrid=tgrid)



        # bounds
        cell_bounds = tgrid.extractCellsBounds()
        cell_0_bounds = cell_bounds[0].__array__().astype('int32')
        cell_1_bounds = cell_bounds[1].__array__().astype('int32')


        lifted_edges = extractLiftedEdges(cell_1_bounds, d=3)

        if self.split == 'train':

            max_l = 3000
            if lifted_edges.shape[0] > max_l + 1:
                indices = numpy.arange(lifted_edges.shape[0])
                numpy.random.shuffle(indices)
                indices=indices[0:max_l]
                lifted_edges = lifted_edges[indices,:]



        #print("local", cell_1_bounds.shape)
        #print("lifted",lifted_edges.shape)



        cellGeometry = tgrid.extractCellsGeometry()
        cell_1_sizes  = cellGeometry[1].sizes().astype('int')
        cell_2_sizes  = cellGeometry[2].sizes().astype('int')

        #print("make gt")

        # transform image level gt to cell1 (sp-boundaries) gt
        cell_1_gt, cell_1_loss_weight, lifted_edge_gt, lifted_edge_loss_weight = self.make_cell_1_gt(
            tgrid=tgrid, 
            cell_1_bounds=cell_1_bounds,
            lifted_edges=lifted_edges,
            sp=sp,
            gt_stack=gt_stack,
            cell_1_sizes=cell_1_sizes,
            cell_2_sizes=cell_2_sizes)
    
        assert lifted_edge_gt.min()>=0.0
        assert lifted_edge_gt.max()<=1.0
        #print("lifted_edge_gt", lifted_edge_gt[0:30])
        #print("lifted_edge_loss_weight", lifted_edge_loss_weight[0:30])


        assert cell_masks.shape[0] ==3
        assert cell_masks.shape[1:3] == pmap.shape 
        assert cell_masks.shape[1:3] == img_raw_big.shape[1:3]


        # REMOVE mean / std / normalize
        pmap -= 0.5
        img_raw_big -= self.bsd500_mean[:, None, None]
        img_raw_big /= self.bsd500_std[:, None, None]

        #print("cell_masks",cell_masks.shape)
        #print("img_raw_big",img_raw_big.shape)
        padded_cell_masks, ps = pad_bsd_to_(cell_masks, ts=1024, mode='constant', channels_front=True)
        padded_image , ps = pad_bsd_to_(img_raw_big,    ts=1024, mode='reflect', channels_front=True)
        padded_pmap  , ps = pad_bsd_to_(pmap,ts=1024, mode='reflect')



        padded_cell_0_coordinates  = cellGeometry[0].__array__()
        padded_cell_0_coordinates += numpy.array(ps, dtype='uint64')
        



        padded_image = numpy.concatenate([padded_image, padded_pmap[None,...]], axis=0)




        if False:
            pylab.imshow(numpy.rollaxis(padded_image[0:3,...],0,3))
            pylab.show()

            pylab.imshow(padded_image[3,...])
            pylab.show()

            pylab.imshow(padded_cell_masks[0,...])
            pylab.show()

            pylab.imshow(padded_cell_masks[1,...])
            pylab.show()

            pylab.imshow(padded_cell_masks[2,...])
            pylab.show()



        c03, c04, fc03, fc04 = nifty.cgp.cell0Cell1Masks(
            padded_image_data=padded_image,
            padded_cell_1_mask=padded_cell_masks[1,...].astype('int'),
            padded_cell_0_coordinates=padded_cell_0_coordinates.astype('int'),
            cell_0_bounds=cell_0_bounds.astype('int'),
            size=16
        )

        

        cell0_3_indices = numpy.where(cell_0_bounds[:,3] == 0)[0]
        if False:
            cell0_4_indices = numpy.where(cell_0_bounds[:,3] != 0)[0]

        # this still starts at 1
        cell0_3_bounds = cell_0_bounds[cell0_3_indices, 0:3]
        if False:
            cell0_4_bounds = cell_0_bounds[cell0_4_indices, :]


        cell0_3_gt, cell0_3_lw = make_cell_0_gt(cell_bounds=cell0_3_bounds,
            cell_1_gt=cell_1_gt,
            jsize=3)

        if False:
            cell0_4_gt, cell0_4_lw = make_cell_0_gt(cell_bounds=cell0_4_bounds,
                cell_1_gt=cell_1_gt,
                jsize=4)



        #print(cell0_3_gt,"braa")
        #print(cell0_4_gt,"braa")

   

        return_dict = OrderedDict()

        ########################################
        # X
        ########################################
        return_dict["padded_image"] = torch.from_numpy(numpy.require(padded_image, dtype='float32',requirements=['C']))
        return_dict["padded_cell_masks"] = torch.from_numpy(numpy.require(padded_cell_masks, requirements=['C'], dtype='int32')).int()
        return_dict["cell_0_bounds"] = torch.from_numpy(numpy.require(cell_0_bounds, requirements=['C']))
        return_dict["cell_1_bounds"] = torch.from_numpy(numpy.require(cell_1_bounds, requirements=['C']))
        return_dict["cell_1_sizes"] = torch.from_numpy(cell_1_sizes).int()
        return_dict["cell_2_sizes"] = torch.from_numpy(cell_2_sizes).int()
        return_dict["cell0_3_bounds"] = torch.from_numpy(cell0_3_bounds).long()
        #return_dict["cell0_4_bounds"] = torch.from_numpy(cell0_4_bounds).long()
        return_dict["c03"] = torch.from_numpy(c03).int()
        #return_dict["c04"] = torch.from_numpy(c04).int()
        return_dict["fc03"] = torch.from_numpy(fc03)
        return_dict["lifted_edges"] = torch.from_numpy(lifted_edges.astype('int')).long()
        
        
        #return_dict["fc04"] = torch.from_numpy(fc04)
        
        ########################################
        # Y
        ########################################
        assert cell0_3_gt.ndim == 1
        #assert cell0_4_gt.ndim == 1

        return_dict["cell_1_gt"] = torch.from_numpy(cell_1_gt)
        return_dict["cell_1_loss_weight"] = torch.from_numpy(cell_1_loss_weight)
        return_dict["cell0_3_gt"] = torch.from_numpy(cell0_3_gt).long()
        #return_dict["cell0_4_gt"] = torch.from_numpy(cell0_4_gt).long()
        return_dict["cell0_3_lw"] = torch.from_numpy(cell0_3_lw)
        #return_dict["cell0_4_lw"] = torch.from_numpy(cell0_4_lw)
        return_dict["lifted_edge_gt"] = torch.from_numpy(lifted_edge_gt)
        return_dict["lifted_edge_loss_weight"] = torch.from_numpy(lifted_edge_loss_weight)
        

        ########################################
        # MORE DATA TO DO ACTUAL PREDICTION
        ########################################
        if return_test_data:
            return_dict["tgrid"] = tgrid
            return_dict["img_raw"] = img_raw
            return_dict["sp"] = sp
            return_dict["image_number"] = self.img_numbers[index]
            return_dict["gt_stack"] = gt_stack 
       

        return return_dict

    def num_inputs(self):
        return 10
    def num_targets(self):
        return 6

    def __len__(self):
        #return 1
        return len(self.img_filenames)

    def _get_bsd_gt_stack(self, filename):
        #from io.cStringIO import StringIO
        import sys,io

        old_stdout = sys.stdout
        sys.stdout = mystdout =  io.StringIO()


    
        gt_stack = vigra.loadBSDGt(filename)

        sys.stdout = old_stdout

        return gt_stack

