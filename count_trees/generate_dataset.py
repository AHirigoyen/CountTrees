"""
Usage:
    generate_dataset <shape_labels> <raster> <output_zip> [--patch_size patch_size --patch_overlap patch_overlap --allow_empty] 

Options:
    --patch_size patch_size         Patch size [default: 400]. 
    --patch_overlap patch_overlap   Patch_overlap (Percentage) [default: 0]. 
    --allow_empty                   If include empty images.
"""
from docopt import docopt
import geopandas as gpd
from shapely.geometry import Polygon
import fiona
from .utils.processing_data import ProcessImages
from deepforest.preprocess import split_raster
from deepforest.utilities import shapefile_to_annotations
import tempfile
import os 
from .utils.zip import zip_folder
import logging
from glob import glob 
import pandas as pd 
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import warnings
warnings.filterwarnings('ignore')


def circle_to_square(circle):
    # Assuming the 'geometry' column contains circular polygons
    try:
        radius = circle.buffer(0).envelope.exterior.xy[0][0] - circle.exterior.xy[0][0]
        side_length = 2 * radius
        centroid = circle.centroid
        min_x, min_y, max_x, max_y = centroid.x - radius, centroid.y - radius, centroid.x + radius, centroid.y + radius
        square = Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])
        return square
    except:
        return None


def generate_squared_shapes(input_shapefile, output_shapefile):
    gdf = gpd.read_file(input_shapefile)
    # Convert circles to squares
    gdf['geometry'] = gdf['geometry'].apply(circle_to_square)
    gdf_clean = gdf.dropna(subset=['geometry'])

    # Reset index, if desired
    gdf_clean.reset_index(drop=True, inplace=True)
    gdf_clean.to_file(output_shapefile)


def main():
    args = docopt(__doc__)
    input_shapefile = args['<shape_labels>'].split(',')
    raster_path = args['<raster>'].split(',')
    output_zip = args['<output_zip>']
    patch_size = int(args['--patch_size'])
    patch_overlap = float(args['--patch_overlap'])
    allow_empty = args['--allow_empty']

    logger.info('Processing shape')

    temdir_save = tempfile.TemporaryDirectory().name
    os.makedirs(temdir_save, exist_ok = True)
    root_output_folder = os.path.join(temdir_save, 'output')
    os.makedirs(root_output_folder, exist_ok = True)

    for i, (input_shapefile, raster_path) in tqdm(enumerate(zip(input_shapefile, raster_path))):

        temdir = tempfile.TemporaryDirectory().name
        os.makedirs(temdir, exist_ok = True)

        output_shapefile = os.path.join(temdir, 'squares.shp')
        generate_squared_shapes(input_shapefile, output_shapefile)

        output_raster = os.path.join(temdir, 'annotations.tif')
        ProcessImages.process_image(raster_path, output_raster)

        df = shapefile_to_annotations(
            shapefile=output_shapefile,
            rgb=output_raster,
            geometry_type="bbox", 
            buffer_size=0.15
        )
                
        temp_csv = os.path.join(temdir, 'annotations.csv')

        df.to_csv(temp_csv, index=False)

        output_folder = os.path.join(root_output_folder, f'output_dir_{i}')

        logger.info('Generate annotations')

        annotations = split_raster(
                path_to_raster=output_raster,
                annotations_file=temp_csv,
                patch_size=patch_size,
                patch_overlap=patch_overlap,
                base_dir=output_folder,
                allow_empty=allow_empty
            )
        

    dfs_paths = glob(os.path.join(root_output_folder, '*', 'annotations.csv'))

    dfs =[]
    for df_path in dfs_paths:
        df = pd.read_csv(df_path)
        name_folder = os.path.dirname(df_path)
        name_folder = os.path.basename(name_folder)
        df['image_path'] = df['image_path'].apply(lambda x: os.path.join(name_folder,x))
        dfs.append(df)

    df = pd.concat(dfs)
    df.to_csv(os.path.join(root_output_folder, 'annotations.csv'), index=False)

    zip_folder(root_output_folder, output_zip)


if __name__ == "__main__":
    main()