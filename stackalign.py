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
# Images that have been cut or grinded from a block are oftentimes not aligned. This IPython notebook uses a fixed target structure in the image (in our case the outline of an overmold) that is visible in all images of the stack to find the best affine transform which aligns all images to the given target. The target is based on one image of the stack where only the fixed structure remains visible and the remaining area is made transparent.

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
from PIL import Image

# %%
from skimage import img_as_float, io

# %%
from stackalign import plot_overlay, PatternFinder, build_stack

# %% [markdown]
# ## Input images, target image, Output dir

# %%
# Load Target File containing the template for the further template matching
target = img_as_float(io.imread("./59-016_links_template-083.png"))
output_dir = Path("./EXPORT")

# Load SVG file containing outline of template and extract path frpom xml format
svg_xml = minidom.parse("./demo.svg")

# Load image collection
ic = io.ImageCollection('./test/*.jpg')

# %% [markdown]
# ## Check overlay and target
#
# The overlay is only used to visualize if and how the target pattern fits to an image.

# %%
#Quick check if the target image and the SVG outline match
svg_path = svg.path.parse_path([path.getAttribute('d') for path in svg_xml.getElementsByTagName('path')][0])
svg_xml.unlink()
overlay = plot_overlay(target, svg_path, figsize=(5,5))
del overlay

# %% [markdown]
# ## Search Strategy

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

# %% [markdown]
# ## Align the stack of images
#
# ... takes some time.

# %% {"outputExpanded": true}
# Assure the border of the target is transparent
target[0,:,3] = 0.0
target[-1,:,3] = 0.0
target[:,0,3] = 0.0
target[:,-1,3] = 0.0

import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
import warnings
# process the stack
with warnings.catch_warnings():
    warnings.simplefilter("once")
    result = build_stack(ic[:],
                         target,
                         rough_search_strategy=rough_search_strategy,
                         fine_search_strategy=fine_search_strategy,
                         write_files=output_dir,
                         svg_path=svg_path,
                         plot='all')
