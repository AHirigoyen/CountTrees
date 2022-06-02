"""Module to process raster data to be compatible with Deepforest.

Processing data to be compatible with Deepforest package in training and
inference. For training the labels are transformed from Arcgis Pro output
dataset to Deepforest format, also for inference and training the raste data
is transformed from float32 to int8.

Usage:
    processing_data.py --input_dir FOLDER --output_dir FOLDER
    processing_data.py --input_raster IMG --output_raster IMG

Options:
    --input_dir FOLDER     Folder with the structure of Arcgis ouput.
    --output_dir FOLDER    Folder to save ouput data.
    --input_raster IMG     Raster image (.tif) with datatype float32
    --output_raster IMG    Name output raster (.tif) with datatype uint8
"""

import os
from glob import glob

from docopt import docopt
from osgeo import gdal
from tqdm import tqdm


class ProcessImages:
    """Process Images a save to directory.
    """
    def __init__(self, output_dirname: str) -> None:
        self._output_dirname = output_dirname

    def __call__(self, path_img: str) -> None:
        output_path_img = os.path.basename(path_img)
        output_path_img = os.path.join(self._output_dirname, output_path_img)
        self.process_image(path_img, output_path_img)

    @staticmethod
    def process_image(path_img: str, output_path_img: str) -> None:
        """ Generate new raster data with datatype uint8 from raster data
        with datatype float32.
        """
        # Opening raster from disk
        raster_data = gdal.Open(path_img)

        # Calculate the mininimun and maximun to Scale bands between 0 and 255
        maxs,mins = [],[]
        for i in range(3):
            srcband = raster_data.GetRasterBand(i+1)
            srcband.ComputeStatistics(0)
            mins.append(srcband.GetMinimum())
            maxs.append(srcband.GetMaximum())
        _min = min(mins)
        _max = max(maxs)

        # Generating new raster data after resclaing
        gdal.Translate(output_path_img, raster_data,
                        scaleParams = [[_min,_max,0,255]],
                        outputType = gdal.GDT_Byte)


class ProcessLabels:
    """ Transform multiples labels from Arcgis Pro format (Kittie) to one csv
    compatible with Deepforest.
    """
    def __init__(self, output_dirname: str) -> None:
        self._name_cvs = os.path.join(output_dirname, 'annotations.csv')
        columns = ['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label']
        with open(self._name_cvs,'w+') as file_csv:
            file_csv.write(",".join(columns))
            file_csv.write("\n")

    def __call__(self, path_annotation: str) -> None:
        image_path =  os.path.basename(path_annotation).replace('txt', 'tif')
        data = []

        with open(path_annotation, "r") as file_annotation:
            with open(self._name_cvs,'a') as file_csv:
                for line in file_annotation:
                    # Extract xmin, ymin, xmax, ymax
                    bbx = line.split()[4:8]

                    # Transflor to int
                    bbx = map(lambda x: float(x.replace(',','.')),bbx)
                    bbx = list(map(int,bbx))

                    # Remove not boxes
                    if ((bbx[2] - bbx[0]) <= 0) or ((bbx[3] - bbx[1]) <= 0):
                        continue

                    data = [image_path,*bbx,'Tree']
                    file_csv.write(",".join(map(str, data)))
                    file_csv.write("\n")


def process_data(intput_dirname: str, output_dirname: str) -> None:
    """Transform raster data and labels from Arcgis Pro to Deepforest
    format.
    """
    process_images = ProcessImages(output_dirname)
    process_labels = ProcessLabels(output_dirname)

    dir_images = os.path.join(intput_dirname, 'images')
    dir_labels = os.path.join(intput_dirname, 'labels')

    os.makedirs(output_dirname, exist_ok = True)

    list_images = glob(os.path.join(dir_images, '*tif'))
    for path_image in tqdm(list_images):
        process_images(path_image)
        name_label = os.path.basename(path_image).replace('tif', 'txt')
        name_label = os.path.join(dir_labels, name_label)
        process_labels(name_label)


if __name__ == "__main__":
    args = docopt(__doc__)

    input_dirname = args['--input_dir']
    out_dirname = args['--output_dir']

    input_raster = args['--input_raster']
    ouput_raster = args['--output_raster']

    if input_dirname: 
        process_data(input_dirname, out_dirname)
    elif input_raster: 
        ProcessImages.process_image(input_raster, ouput_raster)

    
