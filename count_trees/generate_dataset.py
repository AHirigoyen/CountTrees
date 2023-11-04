"""
Usage:
    generate_dataset <shape_labels> <raster> <output_zip>
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


def circle_to_square(circle):
    # Assuming the 'geometry' column contains circular polygons
    radius = circle.buffer(0).envelope.exterior.xy[0][0] - circle.exterior.xy[0][0]
    side_length = 2 * radius
    centroid = circle.centroid
    min_x, min_y, max_x, max_y = centroid.x - radius, centroid.y - radius, centroid.x + radius, centroid.y + radius
    square = Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])
    return square


def generate_squared_shapes(input_shapefile, output_shapefile):
    gdf = gpd.read_file(input_shapefile)
    # Convert circles to squares
    gdf['geometry'] = gdf['geometry'].apply(circle_to_square)
    gdf.to_file(output_shapefile)


def main():
    args = docopt(__doc__)
    input_shapefile = args['<shape_labels>']
    raster_path = args['<raster>']
    output_zip = args['<output_zip>']

    temdir = tempfile.gettempdir()
    output_shapefile = os.path.join(temdir, 'squares.shp')

    generate_squared_shapes(input_shapefile, output_shapefile)


    output_raster = os.path.join(temdir, 'precessed.tif')
    ProcessImages.process_image(raster_path, output_raster)


    df = shapefile_to_annotations(
        shapefile=output_shapefile,
        rgb=output_raster, convert_to_boxes=False, buffer_size=0.15
    )
            
    temp_csv = os.path.join(temdir, 'annotations.csv')

    df.to_csv(temp_csv, index=False)

    annotations = split_raster(
            path_to_raster=output_raster,
            annotations_file=temp_csv,
            patch_size=450,
            patch_overlap=0,
            base_dir=output_folder,
            allow_empty=False
        )
    
    zip_folder(root_path, output_zip)


if __name__ == "__main__":
    main()