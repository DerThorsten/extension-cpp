import torch
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

from model import *
from bsd_ds import *

from predictor import * 


if __name__ == '__main__':


    class LossWrapper(nn.Module):
        def __init__(self):
            super(LossWrapper, self).__init__()
            #self.loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.5, 2.0]), reduce=False)
            self.loss = torch.nn.BCELoss(reduce=False)
            self.loss2 = torch.nn.BCELoss(reduce=False)

        def forward(self,thepred, all_gt):
            cell_1_prediction, j3pred, j4pred = thepred
            gt, sizes, cell0_3_gt, cell0_4_gt = all_gt
            cell_1_prediction = torch.squeeze(cell_1_prediction)

            cell0_3_gt = torch.squeeze(cell0_3_gt)
            cell0_4_gt = torch.squeeze(cell0_4_gt)

            gt = torch.squeeze(gt)
            w = (gt + 1)**2.75
            sizes = torch.squeeze(sizes.float())
            #print('it', cell_1_prediction.size(), cell_1_gt.size())
            
            l =  self.loss(cell_1_prediction, gt)
            l = l * w
            l = torch.sum(l * sizes) / torch.sum(sizes)


            theloss3 = self.loss2(j3pred, cell0_3_gt)

            l30 = theloss3[:,0] * (1.0 + cell0_3_gt[:,0])*2.75
            l31 = theloss3[:,1] * (1.0 + cell0_3_gt[:,1])*2.75
            l32 = theloss3[:,2] * (1.0 + cell0_3_gt[:,2])*2.75

            theloss3 = torch.sum(l30 + l31 + l32) / (3.0*cell0_3_gt.size(0))



            theloss4 = self.loss2(j4pred, cell0_4_gt)
            theloss4 = self.loss2(j4pred, cell0_4_gt)

            l40 = theloss4[:,0] * (1.0 + cell0_4_gt[:,0])*2.75
            l41 = theloss4[:,1] * (1.0 + cell0_4_gt[:,1])*2.75
            l42 = theloss4[:,2] * (1.0 + cell0_4_gt[:,2])*2.75
            l43 = theloss4[:,3] * (1.0 + cell0_4_gt[:,3])*2.75

            theloss4= torch.sum(l40 + l41 + l42 + l43) / (4.0*cell0_4_gt.size(0))
            print()

            lsum  = l + 0.3*theloss3 + 0.1*theloss4

            print("THELOSS: ",float(lsum.item()))
            print("     c1",float(l.item()), "c0_3",float(theloss3.item()), "c0_4",float(theloss4.item()))
            return lsum
    import torch.nn as nn
    from inferno.io.box.cifar import get_cifar10_loaders
    from inferno.trainers.basic import Trainer
    from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
    from inferno.extensions.layers.convolutional import ConvELU2D
    from inferno.extensions.layers.reshape import Flatten

    # Fill these in:
    LOG_DIRECTORY = '/home/tbeier/src/extension-cpp/log/'
    SAVE_DIRECTORY = '/home/tbeier/src/extension-cpp/savedir/'

    USE_CUDA = bool(1)

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Define transforms (1)
    #transforms = transforms.Compose([transforms.CenterCrop(100), transforms.ToTensor()])


    bsd_root = "/home/tbeier/datasets/BSR/BSDS500/"
    pmap_root = "/home/tbeier/src/holy-edge/hed-data/out"
    split = "train"



    # AUGMENTATION
    import inferno.io.transform as trafo

    joint_transformation = trafo.Compose(
        #trafo.image.ElasticTransform(1,1),
        trafo.image.RandomRotate(),
        # trafo.image.RandomTranspose(),
        # trafo.image.RandomFlip(),

    )




    



    bsd_train = Bsd500Sp(
        bsd_root=bsd_root, 
        pmap_root=pmap_root,
        split='train', 
        joint_transformation=joint_transformation)

    bsd_val = Bsd500Sp(
        bsd_root=bsd_root, 
        pmap_root=pmap_root,
        split='val', 
        joint_transformation=None)

    model = ConvNet()#.double()

    #model.double()
    # Build trainer
    trainer = Trainer(model) \
        .build_criterion(LossWrapper) \
        .build_criterion(LossWrapper) \
        .build_optimizer('Adam') \
        .validate_every((4, 'epochs')) \
        .save_every((4, 'epochs')) \
        .save_to_directory(SAVE_DIRECTORY) \
        .set_max_num_epochs(100) \
        # .build_logger(TensorboardLogger(log_scalars_every='never',
        #                             log_images_every='never'), 
        #           log_directory=LOG_DIRECTORY)
    
    # Bind loaders
    train_loader = torch.utils.data.DataLoader(dataset=bsd_train, num_workers=1)
    val_loader = torch.utils.data.DataLoader(dataset=bsd_val, num_workers=0)

    num_inputs = bsd_train.num_inputs()
    num_targets = bsd_train.num_targets()
    
    trainer \
      .bind_loader('train',    train_loader, num_inputs=num_inputs, num_targets=num_targets) \
      .bind_loader('validate', val_loader,   num_inputs=num_inputs, num_targets=num_targets) \



    trainer.cuda()
    trainer.fit()
