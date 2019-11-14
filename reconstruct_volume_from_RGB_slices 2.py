# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernel_info:
#     name: python3
#   kernelspec:
#     display_name: Python (hacking)
#     language: python
#     name: py3
# ---

# %% [markdown]
# # stackalign – Aligning a Stack of Images Based On a Fixed Target Outline

# %% [markdown]
# Authors: Daniel Sieber, Samuel John
#
# #### Abstract
#
# Images that have been cut or grinded from a block are oftentimes not aligned. This IPython notebook uses a fixed target structure in the image (in our case the outline of an overmold) that is visible in all images of the stack to find the best affine transform which aligns all images to the given target. The target is based on one image of the stack where only the fixed structure remains visible and the remaining area is made transparent.
#
# #### Repository
#
# <https://github.com/awesomecodingskills/reconstruct_volume_from_RGB_slices>
#
# #### TODO
#
# - Write better Abstract
# - Add "How to cite" statement and link to paper (DOI) here
# - Improve code commenting
#
#
# #### [The MIT License (MIT)](http://opensource.org/licenses/MIT)
#
# Copyright (c) 2015-2017 Daniel Sieber, Samuel John
#
#
# <div style="font-size:7pt;">
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# </div>

# %% [markdown]
# ### Imports & Set-Up

# %%
# Plot in this IPython Notebook instead of opening separate windows
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
from matplotlib import pyplot as plt

# %%
import logging
from xml.dom import minidom
import svg.path  # you might need to `pip install svg.path`

# %%
from skimage import img_as_float, io

# %%
from stackalign import plot_overlay, PatternFinder, build_stack

# %%
# !pip install git+https://github.com/HearSys/pattern_finder_gpu.git

# %% [markdown]
# ### Start of main script

# %%
# Load Target File containing the template for the further template matching
target = img_as_float(io.imread("/Users/sam/HoerSys/HS03_External_Projects/RESPONSE_tube_2018_Lena/59-016_links_template-083.png"))

# Load SVG file containing outline of template and extract path frpom xml format
svg_xml = minidom.parse("/Users/sam/HoerSys/HS03_External_Projects/RESPONSE_tube_2018_Lena/59-016_links_template-083.svg")
svg_path = svg.path.parse_path([path.getAttribute('d') for path in svg_xml.getElementsByTagName('path')][0])
svg_xml.unlink()

# Load image collection
ic = io.ImageCollection('/Users/sam/HoerSys/HS03_External_Projects/RESPONSE_tube_2018_Lena/59-016 links/*.jpg',conserve_memory=True)
#ic = io.ImageCollection('./test_image.tif')

# Assure the border of the target is transparent
target[0,:,3] = 0.0
target[-1,:,3] = 0.0
target[:,0,3] = 0.0
target[:,-1,3] = 0.0

# %%
#Quick check if the target image and the SVG outline match
overlay = plot_overlay(target, svg_path, figsize=(5,5))
del overlay

# %%
#Definition of search strategy for brute force
rough_search_strategy = [dict(rescale=0.02, angle_range=(0,0,1), roi_hw=(51,51)),
                         dict(rescale=0.08, angle_range=(-25,  25,  9), roi_hw=(61, 61)),
                         dict(rescale=0.1, angle_range=(-4,  4,  17), roi_hw=(61, 61)),
                         dict(rescale=0.15, angle_range=(-0.5,  0.5,  17), roi_hw=(61, 61)),
                         dict(rescale=0.2, angle_range=(-0.1,  .1,  22), roi_hw=(81, 81)),
                      ]

fine_search_strategy = []
# fine_search_strategy = [dict(rescale=0.2, tol=120.0)]

# %%
import warnings

# %%
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

# %% {"outputExpanded": true}
#Execution of image alignment
with warnings.catch_warnings():
    warnings.simplefilter("once")  # strangely "once" does not seem to do what it says... so for now just "shut up"
    result = build_stack(ic[:],
                         target,
                         rough_search_strategy=rough_search_strategy,
                         fine_search_strategy=fine_search_strategy,
                         write_files='./EXPORT',
                         svg_path=svg_path,
                         plot='all')
