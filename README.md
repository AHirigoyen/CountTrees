# Count Trees 

## Install Repository 
```
git clone 
cd CountTrees
pip install --upgrade -r requirements.txt
```

## Processing Dataset for Compatibility in Training  with Deepforest.

Processing of dataset for training from Arcgis Pro Labeled tools to be compatible with Deepforest. 

```
python utils/processing_data.py --input_dit FOLDER --output_dir FOLDER
```


## Training 
The input dir must to be in the format of ```python utils/processing_data.py``` output directory.

```
python train.py --input_dir FOLDER  --output_dir FOLDER 
```

## Inference 

```
python inference.py model_path img output_dir
```