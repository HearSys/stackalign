import sys
import logging

# Import external modules used by this script
from skimage import img_as_float, io, transform

# Scientific Python and typed array/matrix support (by including NumPy)
import scipy as sp

# Parsing svg files and accessing paths in there
from xml.dom import minidom
import svg.path  # you might need to `pip install svg.path`
from PIL import Image
import tqdm
from tqdm.notebook import tqdm
from tqdm import tnrange
import contextlib
from pathlib import Path
from matplotlib import pyplot as plt
from pathlib import Path
from xattr import xattr
from struct import unpack
import sys

from pattern_finder_gpu import center_roi_around, find_pattern_rotated, PatternFinder


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
    if "darwin" in sys.platform:
        return None

    attrs = xattr(path)

    try:
        finder_attrs = attrs[u'com.apple.FinderInfo']
        flags = unpack(32*'B', finder_attrs)
        color = flags[9] >> 1 & 7
    except KeyError:
        color = 0

    return colornames[color]


def print_parameters(T, value=None):
    """
    Function that returns a str with the components an affine transformation matrix `T`.
    Additionally the resulting `error` value can be integrated.
    - `T`: skimage.transform.AffineTransformation object
    """
    return (f" x,y=({T.translation[0]:.0f}, {T.translation[1]:.0f}),"
            f" r={sp.rad2deg(T.rotation):.3f}º,"
            f" scale(x,y)=({T.scale[0]:.3f},{T.scale[1]:.3f}),"
            f" shear={T.shear:.4f} => err={value:.8f}")


def plot_overlay(image, svg_path_or_image, figsize=(15,15), overlay_color='magenta'):
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
    if isinstance(svg_path_or_image, Image.Image):
        ax.imshow(svg_path_or_image)
    else:
        #Sample 10000 points from the path and get their coordinates
        numberSamplePoints = 10000
        overlay_coords = sp.array([svg_path_or_image.point(p/numberSamplePoints) for p in range(numberSamplePoints)])
        #Plot the path
        ax.plot(overlay_coords.real, overlay_coords.imag, color=overlay_color)
    fig.canvas.draw()
    return fig



def build_stack(images, target, rough_search_strategy=None, fine_search_strategy=None,
                plot=False, write_files=False, svg_path=None, PF=None):
    """
    - `images`: ndarray or skimage.io.collection.ImageCollection object containing image to be aligned
    - `target`: ndarray containing outline to be searched
    - `rough_search_strategy`: list of dictionaries containing values for rescale(float between 0 and 1),
                       angle range to be searched ([start,end,no_steps]), and ROI size (height,width)
    - `fine_search_strategy`: list of dictionaries containing the values for the fine tuning optimizer:
        + rescale: The rescale factor (float between 0 and 1) to compute the similarity during optimization.
    - `plot`: Define level of detail for plotting[False,True,'all']
    - `write_files`: Path to where to write the output files.
    - `PF`: PatternFinder instance (optional)
    """

    # Create Patternfinder if none exists
    if PF is None:
        PF = PatternFinder(partitions=10)
        PF.logger.setLevel(logging.INFO)

    if write_files:
        write_files = Path(write_files)

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
        imagelist = sorted(set(map(Path, images.files)))
    else:
        imagelist = map(Path, images)

    # Redirect stdout to tqdm.write() (don't forget the `as save_stdout`)
    with std_out_err_redirect_tqdm() as orig_stdout:
        # tqdm needs the original stdout
        # and dynamic_ncols=True to autodetect console width
        for im_nr, image_file in tqdm(enumerate(imagelist), desc='overall', file=orig_stdout, dynamic_ncols=True):
            out_file_name = write_files / f"{image_file.stem}.png"
            out_plotfile_name = write_files / f"Plot_{image_file.stem}.png"
            out_logfile = write_files / f"{image_file.stem }.log"

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
                    print(f"\n --- Round {i+1} ---")
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
                overlay.savefig(write_files / f"Plot_{image_file.stem}.png", dpi=100)
                sp.savetxt(write_files / f"Transform_{image_file.stem}.csv",
                            trans.params, fmt='%.5f', delimiter=';' )

            if plot:
                plt.close(overlay)

            logging.getLogger().removeHandler(fh)

    return final_transforms


def align_image_brute_force(image, target, search_strategy, plot=False, write_files=False, PF=None):
    if PF is None:
        PF = PatternFinder(partitions=10)

    target_center = sp.array(target.shape[:2]) / 2. - 0.5
    im_center = sp.array(image.shape[:2]) / 2. - 0.5

    #Initialize transformation between image and target as identity
    T = transform.AffineTransform(matrix=sp.array([[1,0,0],[0,1,0],[0,0,1]]))
    best_value = None

    logger = logging.getLogger('stackalign')

    for nr, search_phase in enumerate(search_strategy):
        logger.info("\nSearch phase {0}".format(nr))
        best_angle =  sp.rad2deg(T.rotation)
        angle_range = sp.linspace(
            search_phase["angle_range"][0] - best_angle,
            search_phase["angle_range"][1] - best_angle,
            search_phase["angle_range"][2]
        )

        best_coord = sp.array([int(im_center[0]+T.translation[0]),
                               int(im_center[1]+T.translation[1])])

        logger.debug(f"best so far: x,y=({best_coord[0]},{best_coord[1]}), r={best_angle:0.3f}º")
        T, value = find_pattern_rotated(PF, target, image,
                                       rescale=search_phase["rescale"],
                                       rotations=angle_range,
                                       roi_center_rc=best_coord,
                                       roi_size_hw=search_phase["roi_hw"],
                                       plot=plot,
                                       progress=tqdm)

        # TODO: Check if this can be done more efficiently
        # image_rescaled = transform.rescale(image,search_phase["rescale"])
        # Print parameters
        logger.info(print_parameters(T, value))

    return T, value


def loss_fcn(guess, PF, target_scaled, image_scaled, rescale, plot):

    # T = transform.AffineTransform (rotation=guess[2],translation=[guess[0],guess[1]])
    T = transform.AffineTransform (rotation=guess[2],shear=guess[5],
                                   scale=[guess[3],guess[4]],translation=[guess[0],guess[1]])


    scale_mat = sp.asarray(transform.AffineTransform(scale=[rescale, rescale]).params)
    combined_transform = scale_mat * T.params * sp.linalg.inv(scale_mat)

    # Create "fake" ROI around image center with size one
    roi_center = sp.array(image_scaled.shape[:2])/2.0 - 0.5
    roi = center_roi_around(roi_center, [1,1])

    # Execute Pattern Finder and calculate best match
    transformed_targed = transform.warp(target_scaled,
                                        sp.linalg.inv(combined_transform),
                                        output_shape=image_scaled.shape[:2])
    PF.set_pattern(transformed_targed/transformed_targed.max())
    out, min_coords, value = PF.find(roi=roi)

    logger = logging.getLogger('stackalign')
    logger.info(print_parameters(T, value))

    return value


def align_image_local_optim(image, target, T, PF=None, plot=False, **kws):

    rescale = kws.pop("rescale", 1)  # Extract and remove "rescale" from kws and if not in there, default to 1

    if PF is None:
        PF = PatternFinder(partitions=10)

    # Convert initialGuess transformation matrix into an ndarray with six entries for the DOFs
    # initialGuess = sp.asarray([sp.asscalar(T.translation[0]),
    #                            sp.asscalar(T.translation[1]),
    #                            T.rotation])
    initialGuess = sp.asarray([sp.asscalar(T.translation[0]),
                               sp.asscalar(T.translation[1]),
                               T.rotation,T.scale[0],T.scale[1],T.shear])

    target_scaled = transform.rescale(target, rescale)
    im_scaled = transform.rescale(image, rescale)

    # Set (and upload to GPU) the image already now,
    # because during optimization it is not changed at all.
    PF.set_image(im_scaled)

#     if plot==True:
    logger = logging.getLogger('stackalign')
    #Calculate normalized error
    logger.info(print_parameters(T,
                                 loss_fcn(initialGuess, PF, target_scaled, im_scaled, rescale, plot)))

    res = sp.optimize.minimize(loss_fcn,
                               initialGuess,
                               args=(PF, target_scaled, im_scaled, rescale, plot),
                               method='BFGS',
                               **kws)

    final_trans = transform.AffineTransform (rotation=res.x[2],shear=res.x[5],
                                             scale=[res.x[3],res.x[4]],translation=[res.x[0],res.x[1]])
    # final_trans = transform.AffineTransform (rotation=res.x[2], translation=[res.x[0],res.x[1]])


    logger.info(print_parameters(final_trans, res.fun))

    return final_trans, res

