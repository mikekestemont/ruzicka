#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

# create directory to write away figures:
img_dir = '../../output/'
if not os.path.isdir(img_dir):
    os.mkdir(img_dir)

# create directory to write away score tables:
table_dir = '../../output/tables/'
if not os.path.isdir(table_dir):
    os.mkdir(table_dir)