# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.1
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

# %%
import logging

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
def print_parameters(T,value=None,nr_pixel=1):
    """
    Function that prints the components an affine transformation matrix on screen.
    Additionally the resulting `error` can be printed in a normalized way.
    (Kind of the average error per pixel to make different rescales comparable)
    - `T`: skimage.transform.AffineTransformation object
    - `value`: sum of distances of pixels between image and target in RGB space
  
    
    Meaning of outputs printed:
        
        x,y: Translations in X,Y direction
        r: Rotation in degrees
        sx,sy: Scale in X,Y direction
        shear: Shear
        value: Normalized error(Kind of the average error per pixel to make different rescales comparable)
    """
    x = T.translation[0]
    y = T.translation[1]
    rot = sp.rad2deg(T.rotation)
    sx = T.scale[0]
    sy = T.scale[1]
    shear = T.shear
    n_error = (value/nr_pixel)-1
    
    return f" x,y=({x:.0f}, {y:.0f}), r={rot:.3f}º, scale(x,y)=({sx:.3f},{sy:.3f}), shear={shear:.4f} => err={n_error:.8f}"


# %%
from tqdm import tnrange, tqdm_notebook
from pathlib import Path
from time import sleep
import contextlib
import sys
from tqdm import tqdm

class DummyTqdmFile(object):
    """Dummy file-like that will write to tqdm"""
    file = None
    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()

@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err



# %%
from xattr import xattr
from struct import unpack


# %%

# %%
def finder_tag_color(path):
    colornames = {
        0: 'none',
        1: 'gray',
        2: 'green',
        3: 'purple',
        4: 'blue',
        5: 'yellow',
        6: 'red',
        7: 'orange',
    }

    attrs = xattr(path)

    try:
        finder_attrs = attrs[u'com.apple.FinderInfo']
        flags = unpack(32*'B', finder_attrs)
        color = flags[9] >> 1 & 7
    except KeyError:
        color = 0

    return colornames[color]


# %%
def build_stack(images, target, rough_search_strategy=None, fine_search_strategy=None,
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

    # Redirect stdout to tqdm.write() (don't forget the `as save_stdout`)
    with std_out_err_redirect_tqdm() as orig_stdout:
        # tqdm needs the original stdout
        # and dynamic_ncols=True to autodetect console width
        for im_nr, image_file in tqdm_notebook(enumerate(imagelist), desc='overall', file=orig_stdout, dynamic_ncols=True):
            out_file_name = write_files + os.sep + os.path.basename(image_file)[0:3] + ".PNG"
            out_plotfile_name = write_files + os.sep + 'Plot_' + os.path.basename(image_file)[0:3] + ".PNG"
            out_logfile = write_files + os.sep + os.path.basename(image_file)[0:3] + ".log"
            
            if Path(out_file_name).exists() and not finder_tag_color(out_plotfile_name) == 'blue':
                print("Skipping (already processed) image Nr. {0} - {1}".format(im_nr, image_file))
                continue

            
                
            # create logger with 'spam_application'
            logger = logging.getLogger('stackalign')
            logger.setLevel(logging.DEBUG)
            # create file handler which logs even debug messages
            fh = logging.FileHandler(out_logfile)
            fh.setLevel(logging.DEBUG)
            logging.getLogger().addHandler(fh)



            if use_ic:
                im = img_as_float(images.load_func(image_file))
                print("\n\nImage Nr. {0} - {1}".format(im_nr, image_file))
            else:
                im = img_as_float(image_file)
                print("\n\nImage Nr. {0}".format(im_nr))

            print("\n === BRUTE FORCE ALIGNMENT ===", flush=True)

            search_strategy = rough_search_strategy
            rough_trans, value = align_image_brute_force(im, target, search_strategy, plot, write_files, PF)

            if plot == 'all':
                im_trans = transform.warp(im, rough_trans, output_shape=[target.shape[0], target.shape[1]])
                overlay = plot_overlay(im_trans, svg_path)
                plt.close(overlay)

            trans = rough_trans

            if fine_search_strategy:
                print("\n === LOCAL OPTIMIZATION ===")

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
            if plot:
                plt.show()

            if write_files:
                io.imsave(out_file_name, im_trans)
                overlay.savefig(write_files + os.sep + "Plot_" + os.path.basename(image_file)[0:3] + ".PNG", dpi=100)
                sp.savetxt(write_files + os.sep + "Transform_" + os.path.basename(image_file)[0:3] + ".CSV", 
                            trans.params, fmt='%.5f', delimiter=';' )
                
            if plot:
                plt.close(overlay)

            logging.getLogger().removeHandler(fh)
            
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
    
    logger = logging.getLogger('stackalign')

    for nr, search_phase in enumerate(search_strategy):
        logger.info("\nSearch phase {0}".format(nr))
        best_angle =  sp.rad2deg(T.rotation)
        angle_range = (search_phase["angle_range"][0] - best_angle,
                       search_phase["angle_range"][1] - best_angle,
                       search_phase["angle_range"][2])

        # Todo: here may be a bug: It seems to me that the best_coord from the last search_phase is not 
        # correctly taken. It should (as it is for the angle) be be at the center of the newly created ROI
        # perhaps it is just the case that the translation has to be corrected for the scale
        best_coord = sp.array([int(im_center[0]+T.translation[0]),
                               int(im_center[1]+T.translation[1])])

        logger.debug(f"best so far: x,y=({best_coord[0]},{best_coord[1]}), r={best_angle:0.3f}º")
        T, value = find_pattern_rotated(PF, target, image,
                                       rescale=search_phase["rescale"],
                                       rotate=angle_range,
                                       roi_center=best_coord,
                                       roi_size=search_phase["roi_hw"], 
                                       plot=plot,
                                       progress=tqdm_notebook)

        # TODO: Check if this can be done more efficiently
        # image_rescaled = transform.rescale(image,search_phase["rescale"])
        # Print parameters
        nr_pixel = search_phase["rescale"]**2 * (image.shape[0] * image.shape[1])
        logger.info(print_parameters(T, value, nr_pixel))
            
    return T, value


# %%
def align_image_local_optim(image, target, T, PF=None, plot=False, **kws):
    
    rescale = kws.pop("rescale", 1)  # Extract and remove "rescale" from kws and if not in there, default to 1
    
    if PF is None:
        PF = PatternFinder(partitions=10)
    
    # Convert initialGuess transformation matrix into an ndarray with six entries for the DOFs
    initialGuess = sp.asarray([sp.asscalar(T.translation[0]),
                               sp.asscalar(T.translation[1]),
                               T.rotation])
#     initialGuess = sp.asarray([sp.asscalar(T.translation[0]),
#                                sp.asscalar(T.translation[1]),
#                                T.rotation,T.scale[0],T.scale[1],T.shear])
    
    target_scaled = transform.rescale(target, rescale)
    im_scaled = transform.rescale(image, rescale)
    nr_pixel = im_scaled.shape[0] * im_scaled.shape[1]
    
    # Set (and upload to GPU) the image already now,
    # because during optimization it is not changed at all.
    PF.set_image(im_scaled)

#     if plot==True:
    logger = logging.getLogger('stackalign')
    #Calculate normalized error
    logger.info(print_parameters(T,
                                 loss_fcn(initialGuess, PF, target_scaled, im_scaled, rescale, plot),
                                 nr_pixel))
    
    res = sp.optimize.minimize(loss_fcn,
                               initialGuess,
                               args=(PF, target_scaled, im_scaled, rescale, plot),
                               method='BFGS',
                               **kws)
    
#     final_trans = transform.AffineTransform (rotation=res.x[2],shear=res.x[5],
#                                              scale=[res.x[3],res.x[4]],translation=[res.x[0],res.x[1]])
    final_trans = transform.AffineTransform (rotation=res.x[2],translation=[res.x[0],res.x[1]])


    logger.info(print_parameters(final_trans, res.fun, nr_pixel))

    return final_trans, res


# %%

def loss_fcn(guess, PF, target_scaled, image_scaled, rescale, plot):
    
    T = transform.AffineTransform (rotation=guess[2],translation=[guess[0],guess[1]])
#     T = transform.AffineTransform (rotation=guess[2],shear=guess[5],
#                                    scale=[guess[3],guess[4]],translation=[guess[0],guess[1]])


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

    logger = logging.getLogger('stackalign')
    nr_pixel = transformed_targed.shape[0] * transformed_targed.shape[1]
    logger.info(print_parameters(T, value, nr_pixel))

    return value

# %% [markdown]
# ### Start of main script

# %%
# Load Target File containing the template for the further template matching
target = img_as_float(io.imread("./target_Probe_Felsenbein_links_100.png"))
# Load SVG file containing outline of template and extract path frpom xml format
svg_xml = minidom.parse("./outline_Probe_Felsenbein_links_100.svg")
svg_path = svg.path.parse_path([path.getAttribute('d') for path in svg_xml.getElementsByTagName('path')][0])
svg_xml.unlink()
# Load image collection
ic = io.ImageCollection('./Probe Felsenbein links/*.tif:./Probe Felsenbein links/*.jpg',conserve_memory=True)
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
rough_search_strategy = [dict(rescale=0.02, angle_range=(0,0,1), roi_hw=(211,211)),
                         dict(rescale=0.08, angle_range=(-25,  25,  62), roi_hw=(61, 61)),
                         dict(rescale=0.1, angle_range=(-3,  3,  32), roi_hw=(161, 161)),
                         dict(rescale=0.15, angle_range=(-0.5,  0.5,  32), roi_hw=(41, 41)),
                         dict(rescale=0.2, angle_range=(-0.1,  .1,  22), roi_hw=(21, 21)),
                      ]


fine_search_strategy = []
# fine_search_strategy = [dict(rescale=0.2, tol=120.0)]

# %%
import warnings

# %% {"outputExpanded": true}
#Execution of image alignment
with warnings.catch_warnings():
    PF=pattern_finder_gpu.PatternFinder(partitions=10)
    PF.logger.setLevel(logging.INFO)
    warnings.simplefilter("ignore")  # strangely "once" does not seem to do what it says... so for now just "shut up"
    result = build_stack(ic[:],
                         target,
                         rough_search_strategy=rough_search_strategy,
                         fine_search_strategy=fine_search_strategy,
                         PF=PF,
                         write_files='./EXPORT',
                         plot=False)

# %%
