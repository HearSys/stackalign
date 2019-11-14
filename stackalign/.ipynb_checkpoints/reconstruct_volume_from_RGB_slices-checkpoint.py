# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Build Stack – Aligning a Stack of Images Based On a Fixed Target Outline

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
# Copyright (c) 2015 Daniel Sieber, Samuel John
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

# %%
import os
import time

# Import external modules used by this script
from skimage import img_as_float, io, transform

# Scientific Python and typed array/matrix support (by including NumPy)
import scipy as sp

# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Write Python objects to disk.
# TODO: This should be replaced by some HDF5 files that store the transformation matrix
import pickle

# Parsing svg files and accessing paths in there
from xml.dom import minidom
import svg.path  # you might need to `pip install svg.path`

# %% [markdown]
# Our own modules:

# %%
import pattern_finder_gpu
from pattern_finder_gpu import center_roi_around, find_pattern_rotated


# %% [markdown]
# ### Definition of functions used by this script:

# %%
def plot_overlay(image, svg_path, figsize=(15,15), overlay_color='magenta'):
    """
    This function plots a path from an SVG_xml and shows it on top of image.
    - `image`: ndarray
    - `svg_path`: svg path object, see `svg.path`
    - `ax`: Matplotlib axes object
    - `figsize`: size of figure in inch (see Matplotlib)
    """
    
    # Create new figure and axes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0,0,1,1])
    #Show transformed image
    ax.imshow(image, interpolation='nearest')
    #Sample 10000 points from the path and get their coordinates
    numberSamplePoints = 10000
    overlay_coords = sp.array([svg_path.point(p/numberSamplePoints) for p in range(numberSamplePoints)])
    #Plot the path
    ax.plot(overlay_coords.real, overlay_coords.imag, color=overlay_color)
    fig.canvas.draw()
    return fig


# %%
def print_parameters(T,value=None,im_scaled=None, end="\n"):
    """
    Function that prints the components an affine transformation matrix on screen.
    Additionally the resulting `error` can be printed in a normalized way.
    (Kind of the average error per pixel to make different rescales comparable)
    - `T`: skimage.transform.AffineTransformation object
    - `value`: sum of distances of pixels between image and target in RGB space
    - `im_scaled`: rescaled ndarray containing image to determine number of pixels
  
    
    Meaning of outputs printed:
        
        x,y: Translations in X,Y direction
        r: Rotation in degrees
        sx,sy: Scale in X,Y direction
        shear: Shear
        value: Normalized error(Kind of the average error per pixel to make different rescales comparable)
    """
    #Calculate normalized error
    norm_value= value / (im_scaled.shape[0]*im_scaled.shape[1])
    
    print(" x = {x:.0f} y = {y:.0f} r = {rot:.3f}º sx = {sx:.3f} sy = {sy:.3f} shear = {shear:.4f} =>value ={n_error:.8f}"
          .format(x=sp.float64(T.translation[0]),
                  y=sp.float64(T.translation[1]),
                  rot=sp.rad2deg(T.rotation),
                  sx=T.scale[0],
                  sy=T.scale[1], 
                  shear=T.shear,
                  n_error=norm_value),
          end=end)


# %%
def build_stack(images, target, rough_search_strategy, fine_search_strategy,
                plot=False, write_files=False, PF=None):
    """
    - `images`: ndarray or skimage.io.collection.ImageCollection object containing image to be aligned
    - `target`: ndarray containing outline to be searched
    - `rough_search_strategy`: list of dictionaries containing values for rescale(float between 0 and 1),
                       angle range to be searched ([start,end,no_steps]), and ROI size (height,width)
    - `fine_search_strategy`: list of dictionaries containing the values for the fine tuning optimizer:
        + rescale: The rescale factor (float between 0 and 1) to compute the similarity during optimization.
    - `plot`: Define level of detail for plotting[False,True,'all']
    - `write_files`: boolean to indicate whether results are saved to file
    - `PF`: PatternFinder instance (optional)
    """    

    # Create Patternfinder if none exists
    if PF is None:
        PF = PatternFinder(partitions=10)
 
    # Initialize list which will later on contain transformations for all files in "images"
    final_transforms = []
    
    # Check whether the input is an ImageCollection
    use_ic = False
    if type(images) is io.ImageCollection:
        use_ic = True       
        # Some tif images contain actually two images (a big one and a smaller
        # thumbnail preview). image_collection therefore seems to generate two
        # entries for each of the files. The load_func, however, always loads
        # the big one, which is then actaully loaded twice. So we use a `set`
        # to make this unique and drop duplicates.
        imagelist = sorted(set(images.files))
    else:
        imagelist = images

    for im_nr, image_file in enumerate(imagelist):
        if use_ic:
            im = img_as_float(images.load_func(image_file))
            print("\n\nImage Nr. {0} {1}".format(im_nr, image_file))
        else:
            im = img_as_float(image_file)
            print("\n\nImage Nr. {0}".format(im_nr))
            
        print("\n === BRUTE FORCE ALIGNMENT ===", flush=True)
        rough_trans, value = align_image_brute_force(im, target, rough_search_strategy, plot, write_files, PF)

        if plot == 'all':
            im_trans = transform.warp(im, rough_trans, output_shape=[target.shape[0], target.shape[1]])
            overlay = plot_overlay(im_trans, svg_path)
            plt.close(overlay)
        
        print("\n === LOCAL OPTIMIZATION ===")

        trans = rough_trans
        for i, strategy in enumerate(fine_search_strategy):
            print("\n --- Round {i} ---".format(i=i+1))
            print("    strategy = {}".format(strategy), flush=True)
         
            # Update the refined `trans` for each round in this search strategy
            trans, res = align_image_local_optim(im, target, trans,
                                                 PF=PF, plot=plot, **strategy)
            # Print parameters of local optimization
            print(res.message, flush=True)            

        final_transforms.append(trans)
        im_trans = transform.warp(im, trans, output_shape=[target.shape[0], target.shape[1]])
        overlay = plot_overlay(im_trans, svg_path)
        plt.show()
                
        if write_files:
            io.imsave(write_files + os.sep + os.path.basename(image_file)[0:3] + ".PNG", im_trans)
            overlay.savefig(write_files + os.sep + "Plot_" + os.path.basename(image_file)[0:3] + ".PNG", dpi=100)
            sp.savetxt(write_files + os.sep + "Transform_" + os.path.basename(image_file)[0:3] + ".CSV", 
                        trans.params, fmt='%.50f', delimiter=';' )

        plt.close(overlay)

    return final_transforms


# %%
def align_image_brute_force(image, target, search_strategy, plot=False, write_files=False, PF=None):
    if PF is None:
        PF = PatternFinder(partitions=10)
    
    target_center = sp.array(target.shape[:2]) / 2. - 0.5
    im_center = sp.array(image.shape[:2]) / 2. - 0.5
    
    #Initialize transformation between image and target as identity
    T = transform.AffineTransform(matrix=sp.asmatrix([[1,0,0],[0,1,0],[0,0,1]]))
    best_value = None
    

    for nr, search_phase in enumerate(search_strategy):
        print("\nSearch phase {0}".format(nr), flush=True)
        best_angle = sp.rad2deg(T.rotation)
        angle_range = (search_phase["angle_range"][0] + best_angle,
                       search_phase["angle_range"][1] + best_angle,
                       search_phase["angle_range"][2])
        best_coord = sp.array([int(im_center[0]+T.translation[0]),int(im_center[1]+T.translation[1])])
        
        T,value = find_pattern_rotated(PF, target, image,
                                       rescale=search_phase["rescale"],
                                       rotate=angle_range,
                                       roi_center=best_coord,
                                       roi_size=search_phase["roi_hw"], 
                                       plot=plot)

        if plot:
            # TODO: Check if this can be done more efficiently
            image_rescaled = transform.rescale(image,search_phase["rescale"])
            # Print parameters
            print_parameters(T, value,image_rescaled)
            #DEBUGGING:
            #print ("=> Value:", value)
            #DEBUGGING:
            #print (T.params)
            
    return T, value


# %%
def align_image_local_optim(image, target, T, PF=None, plot=False, **kws):
    
    rescale = kws.pop("rescale", 1)  # Extract and remove "rescale" from kws and if not in there, default to 1
    
    if PF is None:
        PF = PatternFinder(partitions=10)
    
    # Convert initialGuess transformation matrix into an ndarray with six entries for the DOFs
    initialGuess = sp.asarray([sp.asscalar(T.translation[0]),
                               sp.asscalar(T.translation[1]),
                               T.rotation,T.scale[0],T.scale[1],T.shear])
    target_scaled = transform.rescale(target, rescale)
    im_scaled = transform.rescale(image, rescale)

    # Set (and upload to GPU) the image already now,
    # because during optimization it is not changed at all.
    PF.set_image(im_scaled)

    res = sp.optimize.minimize(loss_fcn,
                               initialGuess,
                               args=(PF, target_scaled, im_scaled, rescale, plot), 
                               method='Nelder-Mead',
                               **kws)
    
    final_trans = transform.AffineTransform (rotation=res.x[2],shear=res.x[5],
                                             scale=[res.x[3],res.x[4]],translation=[res.x[0],res.x[1]])
    
    if plot==True:
        print_parameters(final_trans,res.fun,im_scaled)
        
    print()
    return final_trans, res


# %%
def loss_fcn(guess, PF, target_scaled, image_scaled, rescale, plot):
    
    T = transform.AffineTransform (rotation=guess[2],shear=guess[5],
                                   scale=[guess[3],guess[4]],translation=[guess[0],guess[1]])
    #DEBUGGING:
    #print(T.params)
    scale_mat = sp.asmatrix(transform.AffineTransform(scale=[rescale, rescale]).params)
    combined_transform = scale_mat * T.params * scale_mat.I    
        
    # Create "fake" ROI around image center with size one
    roi_center = sp.array(image_scaled.shape[:2])/2.0 - 0.5
    roi = pattern_finder_gpu.center_roi_around(roi_center, [1,1])

    # Execute Pattern Finder and calculate best match
    transformed_targed = transform.warp(target_scaled,
                                        combined_transform.I,
                                        output_shape=image_scaled.shape[:2])
    PF.set_pattern(transformed_targed)
    out, min_coords, value = PF.find(roi=roi)

    if plot=='all':
        print_parameters(T,value,image_scaled)
        #DEBUGGING:
        #print(" => {}".format(value), flush=True)

    return value


# %% [markdown]
# ### Start of main script

# %%
# Load Target File containing the template for the further template matching
target = img_as_float(io.imread("../Eta/Target_ETA_138_40.png"))
# Load SVG file containing outline of template and extract path frpom xml format
svg_xml = minidom.parse("../Eta/Target_ETA_138_40.svg")
svg_path = svg.path.parse_path([path.getAttribute('d') for path in svg_xml.getElementsByTagName('path')][0])
svg_xml.unlink()
# Load image collection
ic = io.ImageCollection('../Eta/*.tif',conserve_memory=True)
# Assure the border of the target is transparent
target[0,:,3] = 0.0
target[-1,:,3] = 0.0
target[:,0,3] = 0.0
target[:,-1,3] = 0.0

# %%
#Quick check if the target image and the SVG outline match
overlay = plot_overlay(target, svg_path, figsize=(7,7))
del overlay

# %%
#Definition of search strategy for brute force
rough_search_strategy = [dict(rescale=0.1, angle_range=(   0,  0,  1), roi_hw=(51, 51)),
                         dict(rescale=0.1, angle_range=( 35, 55, 101), roi_hw=(31, 31))]

fine_search_strategy = [dict(rescale=0.1, tol=10.0),
                        dict(rescale=0.3, tol=10.0)]

# %%
import warnings
import logging

# %%
#Execution of image alignment
with warnings.catch_warnings():
    PF=pattern_finder_gpu.PatternFinder(partitions=10)
    PF.logger.setLevel(logging.INFO)
    warnings.simplefilter("ignore")  # strangely "once" does not seem to do what it says... so for now just "shut up"
    result = build_stack(ic,
                         target,
                         rough_search_strategy=rough_search_strategy,
                         fine_search_strategy=fine_search_strategy,
                         PF=PF,
                         write_files='../EXPORT',
                         plot=False)

# %%
