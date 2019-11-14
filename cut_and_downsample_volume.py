# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python (hacking)
#     language: python
#     name: py3
# ---

# %% [markdown]
# # Cut out a volume of interest

# %% [markdown]
# ### Imports

# %%
from pathlib import Path

# %%
from skimage import io, transform

# %%
from tqdm import tqdm_notebook

# %%
import numpy as np

# %% [markdown]
# ## Define the rectangle to cut out and scaling

# %%
roi_pixel = (slice(500,3200), slice(500,3200))  # Rectangle: top-row, bottom-row, left-column, right-column
rescale = 0.5

# %% [markdown]
# ### Define output dir

# %%
export_dir = Path('./EXPORT')
cutted_for_slicer = export_dir/"_cutted_for_slicer"
cutted_for_slicer.mkdir(exist_ok=True)

# %% [markdown]
# ## Cut out

# %%
for i, image_file in enumerate(tqdm_notebook(list(sorted(export_dir.glob("??? x20*.png")))),
                                       desc='cut and scale down', dynamic_ncols=True):
    image_cutted = io.imread(image_file)[roi_pixel]
    image_scaled = transform.rescale(image_cutted, scale=rescale, anti_aliasing=True, multichannel=True, mode='constant')
    image_normalized = 
    fname = f"cut-{i:04}.png"
    io.imsave(cutted_for_slicer/fname, (image_scaled*255).astype(np.uint8))
    print(f"{image_file} -> {fname}")
#     io.imshow(image_scaled)

