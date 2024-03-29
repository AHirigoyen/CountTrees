from osgeo import gdal
import numpy as np
import shutil
from PIL import Image, ImageOps


def histogram_equalization(array):
    # Compute the histogram
    hist, bins = np.histogram(array.flatten(), bins=256, range=(0, 256))

    # Compute the cumulative distribution function (CDF)
    cdf = hist.cumsum()
    cdf = 255 * cdf / cdf[-1]

    # Use CDF to equalize the array
    equalized_array = np.interp(array.flatten(), bins[:-1], cdf)

    return equalized_array.reshape(array.shape).astype(np.uint8)

# def equalize_and_replace(input_path):
#     # Open the input raster dataset
#     input_raster = gdal.Open(input_path, gdal.GA_Update)
#     if input_raster is None:
#         raise Exception("Failed to open the input raster.")

#     # Loop through each band and perform histogram equalization
#     for band_num in range(1, input_raster.RasterCount + 1):
#         band = input_raster.GetRasterBand(band_num)
#         array = band.ReadAsArray()

#         # Perform histogram equalization on the band
#         equalized_array = histogram_equalization(array)

#         # Write the equalized array back to the band
#         band.WriteArray(equalized_array)

#     # Close the input dataset
#     input_raster = None


def equalize_and_replace(input_path):
    # Open the geo-referenced raster file
    dataset = gdal.Open(input_path, gdal.GA_Update)

    # Read each band and stack them into a 3D numpy array
    bands = [dataset.GetRasterBand(b).ReadAsArray() for b in range(1, 4)]
    rgb_array = np.dstack(bands)

    # Convert the numpy array to a PIL Image
    rgb_image = Image.fromarray(rgb_array, 'RGB')
    
    # Equalize the RGB image
    equalized_image = ImageOps.equalize(rgb_image)

    # Split the equalized image back into its bands
    equalized_bands = equalized_image.split()

    # Write the equalized bands to the output dataset
    for i, new_band in enumerate(equalized_bands,start=1):
        band = dataset.GetRasterBand(i)
        band.WriteArray(np.array(new_band))

    dataset = None


if __name__ == "__main__":
    input_file = "/home/juan/dev/CountTrees/raste.tif"

    # Make a backup of the original file (optional but recommended)
    #shutil.copy(input_file, 'input_backup.tif')

    # Equalize and replace the original file
    equalize_and_replace(input_file)
