# general
import copy
import itertools
import os
import argparse
import numpy as np
from tqdm import tqdm
import random

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet152, ResNet152_Weights

# local files
from data_loader import *
from utils import *
from NeuronCoverage import *

# mutate
import imgaug.augmenters as iaa
import torchattacks
from style_operator import Stylized
from image_transforms import *

# model
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)


class mutation:
    def __init__(self, model, itr_num=20, strategy=1, coverage_metric=0, seed_selection=0, threshold=0.75,
                 step=0.1):
        self.model = model
        self.itr_num = itr_num
        self.strategy = strategy
        self.coverage_metric = coverage_metric
        self.seed_selection = seed_selection
        self.threshold = threshold
        self.step = step
        self.Coverage_table = NeuronCoverage(self.model)
        self.seed_queue = []
        self.generated_test_set = []

    def generate(self, images):
        self.seed_queue.append(images)
        # behavior Oracle
        out_ori = self.model(images).squeeze()
        label_ori = out_ori.argmax()
        for itr in tqdm(range(self.itr_num)):
            queue_size = len(self.seed_queue)
            mutated_img = self.seed_queue[random.randint(queue_size)].detach().clone()
            # mutate
            # 在以下添加新的mutation方法
            mutated_img.detach_()
            # example
            mutated_img = image_rotation(mutated_img, 10)  # 变换
            # behavior Oracle
            out = self.model(mutated_img).squeeze()
            label = out.argmax()
            # coverage Analysis
            old_coverage = self.Coverage_table.neuron_coverage_rate()
            self.Coverage_table.update_coverage(data=mutated_img)
            new_coverage = self.Coverage_table.neuron_coverage_rate()
            if label_ori == label:
                if new_coverage - old_coverage >= 0.01:
                    self.seed_queue.append(mutated_img)
            else:
                self.generated_test_set.append(mutated_img)

        coverage_rate = self.Coverage_table.neuron_coverage_rate()
        return mutated_img, coverage_rate
