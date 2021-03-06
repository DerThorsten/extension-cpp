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
from inferno.trainers.callbacks.scheduling import AutoLR
from inferno.trainers.callbacks.essentials import SaveAtBestValidationScore
from inferno.trainers.callbacks.tqdm import TQDMProgressBar

from model import *
from bsd_ds import *

from predictor import * 


import warnings
from inferno.trainers.callbacks.scheduling import AutoLR
warnings.filterwarnings("ignore")


if __name__ == '__main__':


    laptop = True

    import torch.nn as nn
    from inferno.io.box.cifar import get_cifar10_loaders
    from inferno.trainers.basic import Trainer
    from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
    from inferno.extensions.layers.convolutional import ConvELU2D
    from inferno.extensions.layers.reshape import Flatten

    # Fill these in:
    
    USE_CUDA = bool(1)

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Define transforms (1)
    #transforms = transforms.Compose([transforms.CenterCrop(100), transforms.ToTensor()])


    if laptop==False:
        bsd_root = "/export/home/tbeier/dataset/BSR/BSDS500/"
        pmap_root = "/export/home/tbeier/bsd500_HED/"
        LOG_DIRECTORY = '/export/home/tbeier/src/extension-cpp/log_new/'
        SAVE_DIRECTORY = '/export/home/tbeier/src/extension-cpp/save_fewer/'

    else:
        bsd_root = "/home/tbeier/datasets/BSR/BSDS500/"
        pmap_root = "/home/tbeier/src/holy-edge/hed-data/out/"
        LOG_DIRECTORY = '/home/tbeier/src/extension-cpp/log_new/'
        SAVE_DIRECTORY = '/home/tbeier/src/extension-cpp/save_fewer/'

    split = "train"



    # AUGMENTATION
    import inferno.io.transform as trafo

    joint_transformation = trafo.Compose(
        #trafo.image.RandomScaleSegmentation([0.8, 1.2]),
        trafo.image.RandomRotate(),
        trafo.image.RandomTranspose(),
        trafo.image.RandomFlip()
    )




    



    bsd_train = Bsd500Sp(
        bsd_root=bsd_root, 
        pmap_root=pmap_root,
        split='train', 
        joint_transformation=joint_transformation)

    if True:
        p0,p1 = bsd_train.get_class_p()
    else:
        p0,p1 = (0.86, 0.14)

    print("class p ", p0,p1)

    bsd_val = Bsd500Sp(
        bsd_root=bsd_root, 
        pmap_root=pmap_root,
        split='val', 
        joint_transformation=joint_transformation)



    model = ConvNet()#.double()

    smoothness = 0.001
    trainer = Trainer(model)



    trainer.build_criterion(LossWrapper(p0=p0, p1=p1))
    trainer.build_optimizer('Adam')#, lr=0.0001)
    trainer.validate_every((1, 'epochs'))
    #trainer.save_every((4, 'epochs'))
    trainer.save_to_directory(SAVE_DIRECTORY)
    trainer.set_max_num_epochs(200) 
    trainer.register_callback(SaveAtBestValidationScore(smoothness=smoothness, verbose=True))
    trainer.register_callback(AutoLR(factor=0.5,
                                  patience='1 epochs',
                                  monitor_while='validating',
                                  monitor='validation_loss',
                                  monitor_momentum=smoothness,
                                  consider_improvement_with_respect_to='previous',
                                  verbose=True))

    trainer.register_callback(TQDMProgressBar())


    
    # Bind loaders
    train_loader = torch.utils.data.DataLoader(dataset=bsd_train, num_workers=8)
    val_loader = torch.utils.data.DataLoader(dataset=bsd_val, num_workers=8)

    num_inputs = bsd_train.num_inputs()
    num_targets = bsd_train.num_targets()


    trainer.load()
    trainer\
      .bind_loader('train',    train_loader, num_inputs=num_inputs, num_targets=num_targets) \
      .bind_loader('validate', val_loader,   num_inputs=num_inputs, num_targets=num_targets) \

    trainer.cuda()
    #E
    trainer.fit()
