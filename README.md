# Count Trees 

## Install Repository 
```
!pip install -U git+https://github.com/aguirrejuan/CountTrees.git --quiet 
```

## Processing Dataset for Compatibility in Training  with Deepforest.

Processing of dataset for training from Arcgis Pro Labeled tools to be compatible with Deepforest. 

```
processing_data --input_dit FOLDER --output_dir FOLDER
```


## Training 
The input dir must to be in the format of ```processing_data``` output directory.

```
train --input_dir FOLDER  --output_dir FOLDER 
```

## Inference 

```
inference model_path img output_dir
```