#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 17:14:08 2022

@author: sulagnasarkar
"""

import sys
import os
from tqdm import tqdm
from time import time

import torch
from torch import nn

from dataset import SocialIQ
import model



val_dataset = SocialIQ('/ocean/projects/dmr120014p/sulagna/Social-IQ/code/socialiq/', 'val', mods={'ac', 'v', 't'})
print(val_dataset)