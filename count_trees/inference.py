"""Perform inference over raster data with datatype uint8 using trained model.

Usage:
    inference --patch_size P <model> <img> <outdir> 

Options:
    --patch_size P    Folder with the dataset in format of Deepforest [default: 400]
"""
import warnings

# Filter out the specific RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely.set_operations")

import os

import pandas as pd
import torch
from deepforest import main as main_model
from deepforest.visualize import plot_prediction_dataframe
from docopt import docopt

from .utils.processing_data import ProcessImages
from .utils.convert_csv_to_shape import project


class Inference:
    """Perform inference over raster data with datatype uint8
    """
    def __init__(self, path_model: str, out_dirname: str) -> None:
        self.model = self.load_model(path_model)
        print(self.model.config)
        self.save_dir_img = out_dirname
        self.save_dir_pred_img = os.path.join(out_dirname,'predictions')
        self.results_df = os.path.join(self.save_dir_img, 'results.csv')
        self.shape_dir =  os.path.join(self.save_dir_img, 'shape')
    
        os.makedirs(self.save_dir_img, exist_ok = True)
        os.makedirs(self.save_dir_pred_img, exist_ok = True)


    def __call__(self, path_img: str, patch_size: int = 400) -> None:
        new_path_img = os.path.basename(path_img)
        new_path_img = os.path.join(self.save_dir_img, new_path_img)
        ProcessImages.process_image(path_img, new_path_img)

        dataframe = self.model.predict_tile(new_path_img, return_plot=False, 
                                patch_size=patch_size, patch_overlap=0.25)
        dataframe['image_path'] = os.path.basename(path_img)

        self.results_df = os.path.join(self.save_dir_img, 'results.csv')
        dataframe.to_csv(self.results_df, index_label='id')
        boxes = project(path_img, dataframe)
        boxes.to_file(self.shape_dir, driver='ESRI Shapefile')


    @staticmethod
    def load_model(path_model: str) -> None:
        if torch.cuda.is_available():
            current_device = torch.device("cuda")
        else:
            current_device = torch.device("cpu")
        model = main_model.deepforest()
        model.model.load_state_dict(torch.load(path_model,map_location=current_device))
        model.to("cuda")
        return model 

    def plot_prediction(self,) -> None:
        import PIL.Image
        PIL.Image.MAX_IMAGE_PIXELS = None
        df = pd.read_csv(self.results_df)
        plot_prediction_dataframe(df, root_dir=self.save_dir_img, 
                                   savedir=self.save_dir_pred_img)




def main():
    args = docopt(__doc__)
    path_model = args['<model>']
    path_img = args['<img>']
    out_dirname = args['<outdir>']
    patch_size = int(args['--patch_size'])
    inference = Inference(path_model, out_dirname)
    inference(path_img,patch_size=patch_size)
    inference.plot_prediction()


if __name__ == "__main__":
    main()