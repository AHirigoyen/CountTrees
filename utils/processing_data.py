"""Module to process raster data to be compatible with Deepforest.

Processing data to be compatible with Deepforest package in training and
inference. For training the labels are transformed from Arcgis Pro output
dataset to Deepforest format, also for inference and training the raste data
is transformed from float32 to int8.

Usage:
    processing_data.py --input_zip FOLDER --output_zip FOLDER
    processing_data.py --input_raster IMG --output_raster IMG

Options:
    --input_zip FOLDER     Folder with the structure of Arcgis ouput.
    --output_zip FOLDER    Folder to save ouput data.
    --input_raster IMG     Raster image (.tif) with datatype float32
    --output_raster IMG    Name output raster (.tif) with datatype uint8
"""

import os
from glob import glob

from docopt import docopt
from osgeo import gdal
from tqdm import tqdm
import numpy as np 
import shutil

import tempfile

from pathlib import Path
import zipfile
import os 
from tqdm import tqdm
from glob import glob
from utils.histo_equalizer import equalize_and_replace


def filter_files(x):
    exclude_files = ['xml','tfw']
    for exclue_file in exclude_files:
        if exclue_file in x:
            return False
    return True 
    

def zip_folder(root_path,name_zip):
    base_diname = os.path.basename(root_path)
    dirname = os.path.dirname(root_path)
    files = Path(root_path).glob('**/*')

    files = filter(lambda x: filter_files(str(x)), files)

    with zipfile.ZipFile(name_zip, mode="w") as archive:
        files = tqdm(files)
        for file_path in files: 
            new_name_file = str(file_path).replace(dirname,'')
            files.set_description(f"Adding {new_name_file}")  
            archive.write(file_path, arcname=new_name_file)


def unzip(file_path,destination_path):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(destination_path)


class ProcessImages:
    """Process Images a save to directory.
    """
    def __init__(self, output_dirname: str) -> None:
        self._output_dirname = output_dirname

    def __call__(self, path_img: str) -> None:
        output_path_img = os.path.basename(path_img)
        output_path_img = os.path.join(self._output_dirname, output_path_img)
        return self.process_image(path_img, output_path_img)

    @staticmethod
    def process_image(path_img: str, output_path_img: str) -> bool:
        """ Generate new raster data with datatype uint8 from raster data
        with datatype float32.
        """
        if os.path.exists(output_path_img):
            return True 
        # Opening raster from disk
        raster_data = gdal.Open(path_img)

        # Calculate the mininimun and maximun to Scale bands between 0 and 255
        maxs,mins = [],[]
        seudo_uniques = []
        for i in range(3):
            srcband = raster_data.GetRasterBand(i+1)
            stats = srcband.ComputeStatistics(0)
            _min, _max, mean, _ = stats
            mins.append(_min)
            maxs.append(_max)
            hist = np.array(srcband.GetHistogram(_min, _max, 1000))
            seudo_uniques.append(sum(hist != 0 ))
        
        #print(mins, maxs)
        _min = min(mins)
        _max = max(maxs)
        seudo_uniques = np.array(seudo_uniques)
        if any(seudo_uniques < 10):
            return False 

        # Generating new raster data after resclaing
        gdal.Translate(output_path_img, raster_data,
                        scaleParams = [[_min, _max, 0, 254]],
                        outputType = gdal.GDT_Byte,
                        noData = 255,
                        bandList = [1,2,3])

        equalize_and_replace(output_path_img)
        return True 


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

    def __call__(self, path_annotation: str, image_path: str) -> None:
        
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


def process_data(input_zip: str, output_zip: str) -> None:
    """Transform raster data and labels from Arcgis Pro to Deepforest
    format.
    """
    temdir = tempfile.gettempdir()
    input_dirname = os.path.join(temdir,'temp_input_preprocessing')
    output_dirname = os.path.join(temdir,'temp_output_preprocessing')

    os.makedirs(input_dirname, exist_ok = True)
    os.makedirs(output_dirname, exist_ok = True)

    print(f'unzip {input_zip} ...')
    unzip(input_zip, input_dirname)

    list_dirs = os.listdir(input_dirname)
    if len(list_dirs) == 1:
        input_dirname = os.path.join(input_dirname,list_dirs[0])

    process_images = ProcessImages(output_dirname)
    process_labels = ProcessLabels(output_dirname)

    dir_images = os.path.join(input_dirname, 'images')
    dir_labels = os.path.join(input_dirname, 'labels')

    list_images = glob(os.path.join(dir_images, '*tif'))

    list_images = tqdm(list_images)
    for i, path_image in enumerate(list_images):
        if  i%50==0:
            sub_name = str(i).zfill(6)
            sub_dirname = os.path.join(output_dirname, sub_name)
            os.makedirs(sub_dirname, exist_ok = True)
            process_images._output_dirname = sub_dirname

        list_images.set_postfix({'image ':os.path.split(path_image)[-1]})
        success = process_images(path_image)
        if success:
            name_label = os.path.basename(path_image).replace('tif', 'txt')
            name_label = os.path.join(dir_labels, name_label)
            path_image = os.path.join(sub_name, os.path.basename(path_image))
            process_labels(name_label,path_image)

    zip_folder(output_dirname, output_zip)
    shutil.rmtree(input_dirname)
    shutil.rmtree(output_dirname)


if __name__ == "__main__":
    args = docopt(__doc__)

    input_zip = args['--input_zip']
    out_zip = args['--output_zip']

    input_raster = args['--input_raster']
    ouput_raster = args['--output_raster']

    if input_zip: 
        process_data(input_zip, out_zip)
    elif input_raster: 
        ProcessImages.process_image(input_raster, ouput_raster)