#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script with additional functionsz
@author: russeldaries
"""
import numpy as np

def one_hot_vector(j):

    y = np.zeros((10,1))
    y[j] = 1.0
    return y


