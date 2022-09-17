"""Perform inference over raster data with datatype uint8 using trained model.

Usage:
    inference.py <model> <img> <outdir>
"""
import os

import pandas as pd
import torch
from deepforest import main
from deepforest.visualize import plot_prediction_dataframe
from docopt import docopt

from utils.processing_data import ProcessImages


class Inference:
    """Perform inference over raster data with datatype uint8
    """
    def __init__(self, path_model: str, out_dirname: str) -> None:
        self.model = self.load_model(path_model)
        self.save_dir_img = out_dirname
        self.save_dir_pred_img = os.path.join(out_dirname,'predictions')
        self.results_df = None

        os.makedirs(self.save_dir_img, exist_ok = True)
        os.makedirs(self.save_dir_pred_img, exist_ok = True)

    def __call__(self, path_img: str) -> None:
        new_path_img = os.path.basename(path_img)
        new_path_img = os.path.join(self.save_dir_img, new_path_img)
        ProcessImages.process_image(path_img, new_path_img)

        dataframe = self.model.predict_tile(new_path_img, return_plot=False, 
                                patch_size=400, patch_overlap=0.25)
        dataframe['image_path'] = os.path.basename(path_img)

        if not self.results_df:
            self.results_df = os.path.join(self.save_dir_img, 'results.csv')
            dataframe.to_csv(self.results_df, index_label='id')
        else: 
            dataframe.to_csv(self.results_df, mode='a', header=False)

    @staticmethod
    def load_model(path_model: str) -> None:
        model = main.deepforest()
        model.model.load_state_dict(torch.load(path_model))
        return model 

    def plot_prediction(self,) -> None:
        import PIL.Image
        PIL.Image.MAX_IMAGE_PIXELS = None
        df = pd.read_csv(self.results_df)
        plot_prediction_dataframe(df, root_dir=self.save_dir_img, 
                                   savedir=self.save_dir_pred_img)


if __name__ == "__main__":
    args = docopt(__doc__)
    path_model = args['<model>']
    path_img = args['<img>']
    out_dirname = args['<outdir>']
    inference = Inference(path_model, out_dirname)
    inference(path_img)
    inference.plot_prediction()