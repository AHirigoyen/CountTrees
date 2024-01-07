# Count-Trees 

## Introduction 

Count-Trees is a comprehensive toolkit designed for processing and training artificial intelligence models on RGB float images using [DeepForest](https://deepforest.readthedocs.io/en/latest/index.html). [DeepForest](https://deepforest.readthedocs.io/en/latest/index.html), a powerful python package, specializes in airborne object detection and classification. It is built upon the pytorch-based Retinanet model and has been pre-trained on extensive datasets as detailed in the research article, "[Individual Tree-Crown Detection in RGB Imagery Using Semi-Supervised Deep Learning Neural Networks](https://www.mdpi.com/2072-4292/11/11/1309)."

To enhance the efficacy of the DeepForest pre-trained model, Count-Trees introduces capabilities for model fine-tuning. Additionally, it preprocesses RGB raster images for compatibility with the DeepForest framework. This preprocessing involves converting float images to integer format and adapting labels from ArcGIS format to the required DeepForest format.

## Install Repository 

To install Count-Trees, use the following pip command:

```
!pip install -U git+https://github.com/aguirrejuan/CountTrees.git --quiet 
```

## Processing Dataset for Compatibility in Training  with Deepforest.

Count-Trees facilitates the conversion of datasets from ArcGIS Pro Labeled tools into a format compatible with Deepforest for training purposes.

```
processing_data --input_dit FOLDER --output_dir FOLDER
```


## Training 

For training, ensure the input directory follows the format specified by the processing_data command's output.


```
train --input_dir FOLDER  --output_dir FOLDER 
```

## Inference 
Execute the following command for model inference:
```
inference model_path img output_dir
```

This improved text provides a clearer and more structured overview of the Count-Trees toolkit, enhancing readability and comprehension.