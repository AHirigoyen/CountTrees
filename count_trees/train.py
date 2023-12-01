"""
Usage:
    train --input_zip ZIP_FILE --output_dir FOLDER [--epochs EPOCHS --batch_size BATCH_SIZE --split SPLIT --checkpoint PATH --upsampling --nms_thresh NMS_THRESH --iou_threshold IOU_THRESHOLD --score_thresh SCORE_THRESH --lr LEARNING_RATE]

Options:
    --input_zip ZIP_FILE                 Folder with the dataset in format of Deepforest.
    --output_dir FOLDER                  Folder to save ouput data (model and evaluation results).
    --epochs EPOCHS                      Number of epochs to train [default: 10]
    --batch_size BATCH_SIZE              Size of batch_size [default: 8]
    --split SPLIT                        Percentage of split [default: 0.2]
    --checkpoint PATH                    Path to checkpoint to continue training
    --upsampling                         Make upsampling
    --nms_thresh NMS_THRESH              Nms_thresh [default: 0.05]
    --iou_threshold IOU_THRESHOLD        iou_threshold [default: 0.4]
    --score_thresh SCORE_THRESH          score_thresh [default: 0.1]
    --lr LEARNING_RATE                   Learning rate [default: 0.01]
"""
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely.set_operations")

import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from deepforest import main as main_model
from docopt import docopt
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import tempfile
from datetime import datetime as dt
from .utils.processing_data import unzip
from .utils.augmentation import get_transform
import shutil
import json
import yaml


def save_json(dict_results, path_file):
    # Serializing json
    json_object = json.dumps(dict_results)
    # Writing to sample.json
    with open(path_file, "w") as outfile:
        outfile.write(json_object)


class Training:
    def __init__(self, input_zip: str, 
                        output_dir: str,
                        checkpoint: str = None,
                        split: float=0.2) -> None:

        self.split = split
        self.input_zip = input_zip
        self.temdir = tempfile.TemporaryDirectory().name
        os.makedirs(self.temdir, exist_ok = True)
        input_dir_dataset = os.path.join(self.temdir,'temp_input')
        os.makedirs(input_dir_dataset, exist_ok = True)
        print(f'unzip {input_zip} ...')
        unzip(input_zip, input_dir_dataset)

        list_dirs = os.listdir(input_dir_dataset)
        if len(list_dirs) == 1:
            input_dir_dataset = os.path.join(input_dir_dataset,list_dirs[0])
        
        self.input_dir_dataset = input_dir_dataset
        self.output_dir = output_dir
        self.model = main_model.deepforest(transforms=get_transform)

        if checkpoint:
            self.model.model.load_state_dict(torch.load(checkpoint))
        else:
            self.model.use_release()
        self.model.to("cuda")
        self.train_file, self.validation_file = self.__split_dataset(self.split) 
        os.makedirs(self.output_dir, exist_ok = True)


    def __split_dataset(self, split: float) -> Tuple[str, str]:
        path_csv = os.path.join(self.input_dir_dataset, 'annotations.csv')
        if split:
            train_annotations = pd.read_csv(path_csv)
            image_paths = train_annotations.image_path.unique()
            np.random.seed(42)
            valid_paths = np.random.choice(image_paths, int(len(image_paths)*split))
            valid_annotations = train_annotations.loc[train_annotations.image_path.isin(valid_paths)]
            train_annotations = train_annotations.loc[~train_annotations.image_path.isin(valid_paths)]
            
            train_file= os.path.join(self.input_dir_dataset, "train.csv")
            validation_file= os.path.join(self.input_dir_dataset, "valid.csv")
            
            train_annotations.to_csv(train_file, index=False)
            valid_annotations.to_csv(validation_file, index=False)

            return train_file,validation_file
        else:
            return path_csv, path_csv
    

    def upsampling(self,):
        train = pd.read_csv(self.train_file)

        new_path = os.path.join(self.input_dir_dataset,'upsampling')
        os.makedirs(new_path, exist_ok = True)

        new_train = train.copy()

        images = new_train['image_path'].unique()
        for img in images: 
            image_path = os.path.join(self.input_dir_dataset, img)
            new_image_path = os.path.join(new_path, img)
            shutil.copy(image_path, new_image_path)

        new_train['image_path'] = new_train['image_path'].apply(lambda x: os.path.join('upsampling',x))

        results = pd.concat([train,new_train])

        results.to_csv(self.train_file, index=False)


    def train(self, epochs: int=10, batch_size: int=8, accelerator: str='auto', upsampling=False,
              nms_thresh=0.05, iou_threshold=0.4, score_thresh=0.1,
              learning_rate=0.01, **kwargs): 
        
        if upsampling:
            self.upsampling()

        self.model.config["train"]["epochs"] = epochs
        self.model.config["train"]["csv_file"] = self.train_file
        self.model.config["train"]["lr"] = learning_rate
        self.model.config["train"]["root_dir"] = self.input_dir_dataset
        self.model.config["train"]["augment"] = True
        self.model.config["accelerator"] = accelerator

        self.model.config["validation"]["csv_file"] = self.validation_file
        self.model.config["validation"]["root_dir"] = self.input_dir_dataset
        
        self.model.config['batch_size'] = batch_size
        self.model.config['nms_thresh'] = nms_thresh
        self.model.config['score_thresh'] = score_thresh

        self.model.config["save-snapshot"] = False
        self.model.config["train"]["preload_images"] = False
        #self.model.config["train"]["fast_dev_run"] = True

        self.evaluate(file_pr='results_pr_pretrained.json',
                      file_torchmetrics='results_torchmetrics_pretrained.json',
                      iou_threshold=iou_threshold)

        # self.model.trainer =  Trainer(
        #                               accelerator=self.model.config["accelerator"],
        #                               enable_checkpointing=False,
        #                               max_epochs=self.model.config["train"]["epochs"],
        #                               default_root_dir=self.ouput_dir,
        #                               **kwargs,
        #                             )

        callback = ModelCheckpoint(dirpath=os.path.join(self.output_dir, 'checkpoints'),
                                   monitor='box_recall', 
                                   mode="max",
                                   save_top_k=3,
                                   filename="box_recall-{epoch:02d}-{box_recall:.2f}")

        self.model.create_trainer(
                                  #logger=TensorBoardLogger(save_dir=os.path.join(self.output_dir,'logdir/')), 
                                  #callbacks=[callback],
                                  default_root_dir=self.output_dir)

        self.model.trainer.fit(self.model)

    def save(self,):
        fingerprint = f"{dt.now():%Y_%m_%d_%H_%M_%S}"
        ouput_model_name = os.path.join(self.output_dir, f"checkpoint_{fingerprint}.pl")

        torch.save(self.model.model.state_dict(), ouput_model_name)
        

        yaml_config = os.path.join(self.output_dir, f"config_model_{fingerprint}.yaml")
        with open(yaml_config, "w") as file:
            yaml.dump(self.model.config,file)


    def evaluate(self, file_pr='results_pr.json', file_torchmetrics='results_torchmetrics.json', iou_threshold = 0.4):
        print('Evaluating...')

        results_torchmetrics = self.model.trainer.validate(self.model)

        results_pr = self.model.evaluate(self.validation_file, self.input_dir_dataset, iou_threshold = iou_threshold)
        results_pr["results"] =  results_pr["results"].to_dict(orient="records")
        results_pr['predictions'] = results_pr['predictions'].to_dict(orient="records")
        results_pr['class_recall'] = results_pr['class_recall'].to_dict(orient="records")

        save_json(results_torchmetrics, os.path.join(self.output_dir, file_torchmetrics))
        save_json(results_pr, os.path.join(self.output_dir, file_pr))

        print(f"Box precision {results_pr['box_precision']}")
        print(f"Box Recall {results_pr['box_recall']}")
        print(f"Class Recall {results_pr['class_recall']}")
        print("Results")
        print(results_pr["results"])
        

def main():
    args = docopt(__doc__)
    input_zip = args['--input_zip']
    out_dirname = args['--output_dir']
    epochs = int(args['--epochs'])
    bath_size = int(args['--batch_size'])
    split = float(args['--split'])
    checkpoint = args['--checkpoint']
    upsampling = args['--upsampling']
    nms_thresh = float(args['--nms_thresh'])
    iou_threshold = float(args['--iou_threshold'])
    score_thresh = float(args['--score_thresh'])
    learning_rate = float(args['--lr'])
   
    training = Training(input_zip, out_dirname, checkpoint, split=split)
    training.train(epochs=epochs, batch_size=bath_size,
                   upsampling=upsampling, nms_thresh=nms_thresh,
                   iou_threshold=iou_threshold, score_thresh=score_thresh,
                   learning_rate=learning_rate)
    training.save()
    training.evaluate(iou_threshold=iou_threshold)


if __name__ == "__main__":
    main()