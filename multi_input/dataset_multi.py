import sys
import os
pwd = os.path.dirname(__file__)

insightface_path = os.path.abspath(os.path.join(pwd,'..','..','face_recognition','insightface'))
sys.path.insert(0,os.path.join(insightface_path,'recognition','arcface_torch'))

import dataset as arcface_torch_dataset
from collections import defaultdict
import tqdm
import random
import copy

import numbers
import os
import queue as Queue
import threading
from typing import Iterable

import mxnet as mx
import numpy as np
import torch
from functools import partial
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils.utils_distributed_sampler import DistributedSampler
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn


def get_dataloader(
    root_dir,
    local_rank,
    batch_size,
    dali = False,
    seed = 2048,
    num_workers = 2,
    multi_input = True,
    ) -> Iterable:
    if multi_input:
        train_set = MultiInput_MXFD(root_dir=root_dir, local_rank=local_rank)
        rank, world_size = get_dist_info()
        train_sampler = DistributedSampler(
            train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

        if seed is None:
            init_fn = None
        else:
            init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

        train_loader = arcface_torch_dataset.DataLoaderX(
            local_rank=local_rank,
            dataset=train_set,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=init_fn,
        )
    else:
        train_loader = arcface_torch_dataset.get_dataloadedr(
            root_dir,
            local_rank,
            batch_size,
            dali,
            seed,
            num_workers
        )
    return train_loader


class MultiInput_MXFD(arcface_torch_dataset.MXFaceDataset):
    def __init__(self, root_dir, local_rank):
        super().__init__(root_dir, local_rank)
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5], std=[0.5]),
             ])
        # traverse the whole dataset and get class map
        class_map_path = os.path.join(root_dir,'class_map.npy')
        if os.path.exists(class_map_path):
            self.class_map = np.load(class_map_path,allow_pickle=True).item()
        else:
            self.class_map = defaultdict(list)

            print('traverse first time to get class index....')
            for idx in tqdm.tqdm(self.imgidx):
                s = self.imgrec.read_idx(idx)
                header, img = mx.recordio.unpack(s)
                label = header.label
                if not isinstance(label, numbers.Number):
                    label = label[0]
                label = int(label)
                self.class_map[str(label)].append(idx)

            if local_rank == 0:
                np.save(class_map_path,np.array(dict(self.class_map)))


    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        img1 = mx.image.imdecode(img).asnumpy()
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = int(label)

        if len(self.class_map[str(label)])> 1:
            idx2,idx3 = random.sample(self.class_map[str(label)],2)
        else:
            idx2 = random.choice(self.class_map[str(label)])
            idx3 = idx2
        s = self.imgrec.read_idx(idx2)
        header,img = mx.recordio.unpack(s)
        img2 = mx.image.imdecode(img).asnumpy()            

        s = self.imgrec.read_idx(idx3)
        header,img = mx.recordio.unpack(s)
        img3 = mx.image.imdecode(img).asnumpy()

        """
        sample0 = [img1,np.zeros_like(img1),np.zeros_like(img1)]
        sample1 = [np.zeros_like(img2),img2,np.zeros_like(img2)]
        sample2 = [np.zeros_like(img3),np.zeros_like(img3),img3]
        """
        sample0 = [img1.copy(),img1.copy(),img1.copy()]
        sample1 = [img2.copy(),img2.copy(),img2.copy()]
        sample2 = [img3.copy(),img3.copy(),img3.copy()]
        sample3 = [img1.copy(),img2.copy(),img3.copy()]
        samples = [sample0,sample1,sample2,sample3]

        label = torch.tensor(label, dtype=torch.long)
        if self.transform is not None:
            for i in range(4):
                samples[i] = [self.transform(s) for s in samples[i]]
                samples[i] = torch.cat(tuple(samples[i]),dim=0)
        samples = torch.cat(tuple(samples),dim=0)
        return samples, label

    def __file__(self):
        return len(self.imgidx)


