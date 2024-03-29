"""
https://gist.github.com/bw4sz/e2fff9c9df0ae26bd2bfa8953ec4a24c
"""

import geopandas as gp
import rasterio
import os
import shapely
from rasterio import features
import numpy as np
from shapely.geometry import shape

def shapefile_to_annotations(shapefile, rgb, savedir="."):
    """
    Convert a shapefile of annotations into annotations csv file for DeepForest training and evaluation
    Args:
        shapefile: Path to a shapefile on disk. If a label column is present, it will be used, else all labels are assumed to be "Tree"
        rgb: Path to the RGB image on disk
        savedir: Directory to save csv files
    Returns:
        None: a csv file is written
    """
    #Read shapefile
    gdf = gp.read_file(shapefile)
    
    #get coordinates
    df = gdf.geometry.bounds
    
    #raster bounds
    with rasterio.open(rgb) as src:
        left, bottom, right, top = src.bounds
        
    #Transform project coordinates to image coordinates
    df["tile_xmin"] = df.minx - left
    df["tile_xmin"] = df["tile_xmin"].astype(int)
    
    df["tile_xmax"] = df.maxx - left
    df["tile_xmax"] = df["tile_xmax"].astype(int)
    
    #UTM is given from the top, but origin of an image is top left
    
    df["tile_ymax"] = top - df.miny 
    df["tile_ymax"] = df["tile_ymax"].astype(int)
    
    df["tile_ymin"] = top - df.maxy
    df["tile_ymin"] = df["tile_ymin"].astype(int)    
    
    #Add labels is they exist
    if "label" in gdf.columns:
        df["label"] = gdf["label"]
    else:
        df["label"] = "Tree"
    
    #add filename
    df["image_path"] = os.path.basename(rgb)
    
    #select columns
    result = df[["image_path","tile_xmin","tile_ymin","tile_xmax","tile_ymax","label"]]
    result = result.rename(columns={"tile_xmin":"xmin","tile_ymin":"ymin","tile_xmax":"xmax","tile_ymax":"ymax"})
    image_name = os.path.splitext(os.path.basename(rgb))[0]
    csv_filename = os.path.join(savedir, "{}.csv".format(image_name))
    
    #ensure no zero area polygons due to rounding to pixel size
    result = result[~(result.xmin == result.xmax)]
    result = result[~(result.ymin == result.ymax)]
    
    #write file
    result.to_csv(csv_filename, index=False)
    

def project(raster_path, boxes):
    """
    Convert image coordinates into a geospatial object to overlap with input image. 
    Args:
        raster_path: path to the raster .tif on disk. Assumed to have a valid spatial projection
        boxes: a prediction pandas dataframe from deepforest.predict_tile()
    Returns:
        a geopandas dataframe with predictions in input projection.
    """

    geometries = []
    with rasterio.open(raster_path) as dataset:
        bounds = dataset.bounds
        pixelSizeX, pixelSizeY = dataset.res
        image = dataset.read(1)
        nodata = dataset.nodata
        crs = dataset.crs
        is_valid = (image != nodata).astype(np.uint8)
        for coords, value in features.shapes(is_valid, transform=dataset.transform):
            if value != 0:
                geom = shape(coords)
                geometries.append(geom)

        # Create a GeoDataFrame
        gdf_shapes = gp.GeoDataFrame(geometry=geometries)
        gdf_shapes.crs = crs

    #subtract origin. Recall that numpy origin is top left! Not bottom left.
    boxes["xmin"] = (boxes["xmin"] * pixelSizeX) + bounds.left
    boxes["xmax"] = (boxes["xmax"] * pixelSizeX) + bounds.left
    boxes["ymin"] = bounds.top - (boxes["ymin"] * pixelSizeY) 
    boxes["ymax"] = bounds.top - (boxes["ymax"] * pixelSizeY)
    
    # combine column to a shapely Box() object, save shapefile
    boxes['geometry'] = boxes.apply(lambda x: shapely.geometry.box(x.xmin,x.ymin,x.xmax,x.ymax), axis=1)
    boxes = gp.GeoDataFrame(boxes, geometry='geometry')
    
    boxes.crs = dataset.crs.to_wkt()

    # Perform a spatial join to find intersections
    intersections = gp.sjoin(boxes, gdf_shapes, how="inner", op="intersects")

    # Drop duplicates because the same box might intersect with multiple shapes
    intersections = intersections.drop_duplicates(subset=boxes.index.name)

    #Shapefiles could be written with geopandas boxes.to_file(<filename>, driver='ESRI Shapefile')
    
    return intersections
