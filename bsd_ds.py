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


    def make_gt(self, tgrid, sp, cell_1_bounds, gt_stack):
        cell_1_gt = numpy.zeros(tgrid.numberOfCells[1], dtype='float32')
        for i in range(gt_stack.shape[2]):
            labels = gt_stack[:,:,i]
            labels = numpy.require(labels.view(numpy.ndarray), requirements=['C'])

            overlap = nifty.ground_truth.overlap(segmentation=sp, 
                                           groundTruth=labels)

            cell_1_gt += overlap.differentOverlaps(cell_1_bounds)


        cell_1_gt /= gt_stack.shape[2]
        return cell_1_gt

    def make_cell_masks(self, tgrid):


        ftgrid = nifty.cgp.FilledTopologicalGrid2D(tgrid)


        cell_0_mask = ftgrid.cellMask([True,False,False])
        cell_0_mask[cell_0_mask!=0] -=  ftgrid.cellTypeOffset[0]

        cell_1_mask = ftgrid.cellMask([False,True,False])
        cell_1_mask[cell_1_mask!=0] -=  ftgrid.cellTypeOffset[1]

        cell_2_mask = ftgrid.cellMask([False,False,True])
        cell_2_mask[cell_2_mask!=0] -=  ftgrid.cellTypeOffset[2]


        cell_masks = numpy.concatenate([
            cell_0_mask[None,...],
            cell_1_mask[None,...],
            cell_2_mask[None,...]
        ], axis=0)

        return cell_masks


    def getitemimpl(self, index, return_test_data=False):



        # read rgb and make it big
        img_raw = readImgC(self.img_filenames[index], dtype='float32')
        small_shape = img_raw.shape[0:2]
        img_raw_big = vigra.sampling.resize(vigra.taggedView(img_raw, "xyc"), [2*s -1 for s in small_shape])

        # read  and normalize the pmap
        pmap_filename = os.path.join(self.pmap_folder, self.img_numbers[index] + '.png')
        pmap = 1.0 - readImgC(pmap_filename, dtype='float32')/255.0

        # read gt stack from disc
        gt_filename = os.path.join(self.gt_folder, self.img_numbers[index] + '.mat')
        gt_stack = self._get_bsd_gt_stack(gt_filename)
        gt_stack = numpy.require(gt_stack, requirements=['C'])

        # basic transformations
        # - this can flip / mirror the img/pmap/gt
   




        # 4d images?
        if False:
            #False
            pmap = pmap#[None,...]
            img_raw = numpy.rollaxis(img_raw,2,0)#[...]
            gt_stack = numpy.rollaxis(gt_stack,2,0)# ...]


            print("pre:" )
            print("     pmap",pmap.shape)
            print("     img_raw",img_raw.shape)
            print("     gt_stack",gt_stack.shape)
        

            # for t in self.joint_transformation.transforms:
            #     try:
            #         t.build_random_variables(imshape=pmap.squeeze().shape)
            #     except TypeError:
            #         t.build_random_variables()

            pmap, img_raw,  gt_stack = self.joint_transformation(pmap, img_raw, gt_stack)
            # img_raw = self.joint_transformation(img_raw)
            # gt_stack = self.joint_transformation(gt_stack)




            print("post:" )
            print("     pmap",pmap.shape)
            print("     img_raw",img_raw.shape)
            print("     gt_stack",gt_stack.shape)


            pylab.imshow(pmap[0,0,...])
            pylab.show()

            pylab.imshow(img_raw[0,0,...]/255.0)
            pylab.show()

            pylab.imshow(gt_stack[0,0,...])
            pylab.show()





        sp, img_raw_big = bsd_sp(img_raw, pmap, n_sp = 2000)
        sp = numpy.require(sp.view(numpy.ndarray), requirements=['C'])
        assert sp.shape == small_shape
        assert list(img_raw_big.shape[0:2]) == [2*s-1 for s in small_shape]
        sp = sp.squeeze()
        sp = numpy.require(sp.view(numpy.ndarray), requirements=['C'], dtype='uint64')
        #img_raw_big = numpy.require(img_raw_big.view(numpy.ndarray), requirements=['C'], dtype='float')

    
        tgrid = nifty.cgp.TopologicalGrid2D(sp)
        cell_masks = self.make_cell_masks(tgrid=tgrid)



        # bounds
        cell_bounds = tgrid.extractCellsBounds()
        cell_0_bounds = cell_bounds[0].__array__().astype('int32')
        cell_1_bounds = cell_bounds[1].__array__().astype('int32')


        cellGeometry = tgrid.extractCellsGeometry()
        cell_1_sizes  = cellGeometry[1].sizes().astype('int')
        cell_2_sizes  = cellGeometry[2].sizes().astype('int')



        # transform image level gt to cell1 (sp-boundaries) gt
        cell_1_gt = self.make_gt(tgrid=tgrid, cell_1_bounds=cell_1_bounds,
            sp=sp,
            gt_stack=gt_stack)



        assert cell_masks.shape[0] ==3
        assert cell_masks.shape[1:3] == pmap.shape 
        assert cell_masks.shape[1:3] == img_raw_big.shape[0:2]


        # REMOVE mean / std / normalize
        pmap -= 0.5
        img_raw_big -= self.bsd500_mean[None, None, :]
        img_raw_big /= self.bsd500_std[None, None, :]

        #print("cell_masks",cell_masks.shape)
        #print("img_raw_big",img_raw_big.shape)
        padded_cell_masks, ps = pad_bsd_to_(cell_masks, ts=1024, mode='constant', channels_front=True)
        padded_image , ps = pad_bsd_to_(img_raw_big,    ts=1024, mode='reflect')
        padded_pmap  , ps = pad_bsd_to_(pmap,ts=1024, mode='reflect')



        padded_cell_0_coordinates  = cellGeometry[0].__array__()
        padded_cell_0_coordinates += numpy.array(ps, dtype='uint64')
        


        # for i in range(3):
        #     pylab.imshow(cell_masks[i,...], cmap=nifty.segmentation.randomColormap())
        #     pylab.show()

        #     pylab.imshow(padded_cell_masks[i,...], cmap=nifty.segmentation.randomColormap())
        #     pylab.show()



        padded_image = padded_image[None,...].astype('double')
        padded_image = numpy.rollaxis(padded_image,3,1)
        padded_image = padded_image[0,...]
        padded_image = numpy.concatenate([padded_image, padded_pmap[None,...]], axis=0)



        c03, c04, fc03, fc04 = nifty.cgp.cell0Cell1Masks(
            padded_image_data=padded_image,
            padded_cell_1_mask=padded_cell_masks[1,...].astype('int'),
            padded_cell_0_coordinates=padded_cell_0_coordinates.astype('int'),
            cell_0_bounds=cell_0_bounds.astype('int'),
            size=16
        )

        
        # for i in range(c03.shape[0]):
        #     jc = c03[i,...]
        #     fc = fc03[i,...]
        #     s = jc[0,...] +  jc[1,...] + jc[2,...]
        #     raw = fc[0,...]
        #     raw[s!=0] = 0
        #     pylab.imshow(raw/255.0, cmap='gray')
        #     pylab.show()   

            # pylab.imshow(s*255)
            # pylab.show()   

        # 
        cell0_3_indices = numpy.where(cell_0_bounds[:,3] == 0)[0]
        cell0_4_indices = numpy.where(cell_0_bounds[:,3] != 0)[0]

        # this still starts at 1
        cell0_3_bounds = cell_0_bounds[cell0_3_indices, 0:3]
        cell0_4_bounds = cell_0_bounds[cell0_4_indices, :]


        # fetch the gt
        cell0_3_gt = cell_1_gt[cell0_3_bounds -1]
        cell0_4_gt = cell_1_gt[cell0_4_bounds -1]

        #print(cell0_3_gt,"braa")
        #print(cell0_4_gt,"braa")

        def thenorm(img):
            img2 = img.view(numpy.ndarray).copy().astype('float32')
            print(type(img2),">?@!?s")
            img2 -= img2.min()
            img2 /= img2.max()
            return img2

        import pylab as plt


     

        # rolltheaxis to the numpy format





        # pylab.show()

        #sys.exit(1)
        #print("padded_image",padded_image.shape)
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
        return_dict["cell0_4_bounds"] = torch.from_numpy(cell0_4_bounds).long()
        return_dict["c03"] = torch.from_numpy(c03).int()
        return_dict["c04"] = torch.from_numpy(c04).int()
        return_dict["fc03"] = torch.from_numpy(fc03)
        return_dict["fc04"] = torch.from_numpy(fc04)
        
        ########################################
        # Y
        ########################################
        return_dict["cell_1_gt"] = torch.from_numpy(cell_1_gt)
        return_dict["cell_1_sizes_t"] = torch.from_numpy(cell_1_sizes.astype('float32'))
        return_dict["cell0_3_gt"] = torch.from_numpy(cell0_3_gt)
        return_dict["cell0_4_gt"] = torch.from_numpy(cell0_4_gt)


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
        return 12
    def num_targets(self):
        return 4

    def __len__(self):
        #return 30
        return len(self.img_filenames)

    def _get_bsd_gt_stack(self, filename):
        #from io.cStringIO import StringIO
        import sys,io

        old_stdout = sys.stdout
        sys.stdout = mystdout =  io.StringIO()


    
        gt_stack = vigra.loadBSDGt(filename)

        sys.stdout = old_stdout

        return gt_stack

