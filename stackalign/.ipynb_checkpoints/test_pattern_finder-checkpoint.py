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
# # Debugging/Testing PatternFinder

# %%
# Plot in this IPython Notebook instead of opening separate windows
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
import scipy as sp
from skimage import io

# %%
import pattern_finder_gpu
from pattern_finder_gpu import center_roi_around

# %%
test_image = sp.ones((50,20,3), dtype=sp.float32) -0.4
test_cross_rc = [2, 10]

test_image[test_cross_rc[0]-2:test_cross_rc[0]+3, test_cross_rc[1]] = (1.0, 0, 0)
test_image[test_cross_rc[0], test_cross_rc[1]-2:test_cross_rc[1]+3] = (0, 1.0, 0)
io.imshow(test_image)

# %%
test_image[test_cross_rc]

# %%
test_target = sp.ones((7,11,4), dtype=sp.float32)
test_target_center_rc = sp.array(test_target.shape[:2]) / 2 - 0.5

# boder transparent
test_target[0,:,3] = 0
test_target[-1,:,3] = 0
test_target[:,0,3] = 0
test_target[:,-1,3] = 0
test_target_cross_rc = sp.around(test_target_center_rc).astype(sp.int32)
test_target[test_target_cross_rc[0]-1:test_target_cross_rc[0]+2, test_target_cross_rc[1], :] = (1.0, 0, 0, 1)
test_target[test_target_cross_rc[0], test_target_cross_rc[1]-1:test_target_cross_rc[1]+2, :] = (0, 1.0, 0, 1)
io.imshow(test_target)

# %%
test_roi = center_roi_around( (15,10), (1,1))

# %%
test_roi

# %%
test_PF = pattern_finder_gpu.PatternFinder(partitions=1)

# %%
test_PF.set_image(test_image)
test_PF.set_pattern(test_target)
test_out, test_rc, test_val = test_PF.find(roi=[2,1,11,11])
print(test_val)
print(test_rc)
io.imshow(test_out)

# %%
test_out, test_rc, test_val = test_PF.find()
print(test_val)

# %%
test_out, test_rc, test_val = test_PF.find(test_target, test_image)
print(test_val)

# %%
io.imshow(test_out)

# %%
test_rc

# %%
test_cross_rc

# %%
assert sp.allclose(test_rc, test_cross_rc)

# %%
from skimage import transform

# %%
test_transe = transform.AffineTransform(translation=(test_rc-test_target_center_rc)[::-1])

# %%
test_transe.params

# %%
test_image_transfromed = transform.warp(test_image,
                                        test_transe,
                                        output_shape=[test_target.shape[0], test_target.shape[1]])

# %%
io.imshow(test_image_transfromed)

# %%
import logging
logging.basicConfig()

# %%
found_trans, found_value = pattern_finder_gpu.find_pattern_rotated(test_PF,
                                                                   test_target,
                                                                   test_image,
                                                                   rotate=(-45, 45 , 5),
                                                                   roi_size=(81,81),
                                                                   roi_center=sp.array((5.0,5.0)),
                                                                   plot='all')

# %%
found_trans.params

# %% [markdown]
# ## Test Rotation and Translation

# %% [markdown]
# Let's use a nice image from the data that comes with skimage and show it

# %%
from skimage import data

# %%
test_image2 = data.coffee()

# %%
io.imshow(test_image2)

# %% [markdown]
# Use a cutout of the `test_image2` as the target

# %%
test_image2.shape

# %%
test_target2 = test_image2[50:350,150:450,:]

# %%
io.imshow(test_target2)

# %%
test_image2.shape

# %%
io.imshow(test_image2[::-1,::-1,:])

# %%
test_image2_transform = transform.AffineTransform(rotation=sp.deg2rad(10), translation=(-50,-100))

# %%
test_image2_rotated_translated = transform.warp(test_image2, test_image2_transform)
io.imshow(test_image2_rotated_translated)

# %%
true_reverse_transform = transform.AffineTransform(translation=(50,100)) + transform.AffineTransform(rotation=-sp.deg2rad(10))
true_reversed_test_image2 = transform.warp(test_image2_rotated_translated, true_reverse_transform)
io.imshow(true_reversed_test_image2)

# %%
io.imshow(true_reversed_test_image2[:,:,0]-test_image2[:,:,0], cmap='hot')

# %%
