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

# %%
from scipy import interpolate
from skimage import io, img_as_float, transform
import scipy as sp
import csv
# %matplotlib inline

# %%
def read_stack_csv (csvfile):
# This function reads layer positions from a csv file which has these positions as first entry rowwise
# The layer positions are returned as a list
    csv_pos= []
    with open (csvfile) as file:
        for row in csv.reader(file):
            csv_pos.append(float(row[0]))
    return csv_pos


# %%
def create_hd_structure (csv_pos, targetLayerSize):
#Function to provide all information needed to create a HD image stack:
#Determine for each layer of the upsampled dataset whether or not it has to be interpolated
#Create an array with the following entries:
#Interpolated Image:[Stack_Position[mm],'Interpolated', 'False', 'False', 0 ]
#Real_Image: [Stack_Position[mm], 'Real_Image', Original Image#,Original Image Pos, Pos Error of this slice]

    #Calculate layer number in upsampled dataset for each layer
    layerNum = []
    for pos in csv_pos:
        layerNum.append(round(pos/targetLayerSize))

    #Initialize and build upsampled stack
    hdstack = []
    for i in range(layerNum[-1]+1):
        try:
            layerNum.index(i)
        except ValueError:
            hdstack.append([i*targetLayerSize,'Interpolated',False,False,0]) 
        else:
            hdstack.append([i*targetLayerSize,'Real_Image',layerNum.index(i),
                                        csv_pos[layerNum.index(i)],i*targetLayerSize-csv_pos[layerNum.index(i)]])
    return hdstack


# %%
def load_resized_image (ic, im_nr, targetPixelNumXY):
#Loads an image from an imageCollection and returns a version rezized to the target pixel number
    im = ic.load_func(ic.files[im_nr])
    image_scaled = img_as_float(transform.resize(im, targetPixelNumXY))
    return image_scaled


# %%
def determineError (hdstack):
#Determine maximal deviation of real slices from their actual position
    maxErr = 0.0
    for i in range(len(hdstack)):
        if hdstack[i][1] == 'Real_Image':
            if hdstack[i][4] > maxErr:
                maxErr=hdstack[i][4]
    return maxErr         


# %%
def interpolate_image (ic, hdstack, im_nr, targetPixelNumXY):
#Determine previous and next "real image" for an interpolated image and resize them
#Determine interpolation factor, Calculate interpolated image by liner interpolation

        #Make sure this is not a real image
        assert hdstack[im_nr][1] == 'Interpolated', 'This is not an interpolated image'
        
        #Find and load the last and next image which are real images
        lastRealImage = im_nr
        while hdstack[lastRealImage][1] == 'Interpolated':
            lastRealImage=lastRealImage-1
        lastImageResized = load_resized_image(ic,hdstack[lastRealImage][2], targetPixelNumXY)
        nextRealImage = im_nr
        while hdstack[nextRealImage][1] == 'Interpolated':
            nextRealImage=nextRealImage+1
        nextImageResized = load_resized_image(ic,hdstack[nextRealImage][2], targetPixelNumXY)
        
        #Calculate interpolation factor
        interpolFactor = ((hdstack[im_nr][0]-hdstack[lastRealImage][3]) /
                          (hdstack[nextRealImage][3]-hdstack[lastRealImage][3]))
        
        #Interpolate pixelwise between the last and next real image
        interpolImage = (1-interpolFactor) * lastImageResized + interpolFactor * nextImageResized
        
        return interpolFactor, interpolImage, lastRealImage, nextRealImage, lastImageResized, nextImageResized


# %%
#Definition of Constants
targetPixelNumXY = (792,792)
targetLayerSize = 0.150
csvfile='../Eta/Layer_Positions_ETA.csv'
images=io.ImageCollection('../Eta/EXPORT ETA V3/*.PNG')
savedir= '../Eta/EXPORT ETA V3/Interpol/'

# %%
#Main Script

#Read CSV file
csv_pos = read_stack_csv (csvfile)

#Create HDStack Information
hdstack = create_hd_structure (csv_pos, targetLayerSize)

#Interpolate
for i in range(len(hdstack)):
    if hdstack[i][1] == 'Interpolated':
        interpolFactor, interpolImage,l,n,li,ni = interpolate_image (images, hdstack, i, targetPixelNumXY)
        io.imsave (savedir+'{0:03d}'.format(i)+".PNG",interpolImage)
        print ('Slice number {0:03d} has been interpolated between {1:03d} and {2:03d} \n'.format(i,l,n))
    else:
        image_scaled=load_resized_image(images, hdstack[i][2], targetPixelNumXY)
        io.imsave (savedir+'{0:03d}'.format(i)+".PNG",image_scaled)
        print ('Slice number {0:03d} has been resized and written \n'.format(i))

# %%
a = images[0] * 0.2 + images[1] * 0.8

# %%
