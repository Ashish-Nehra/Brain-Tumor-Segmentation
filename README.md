# Brain-Tumor-Segmentation

Use google colab for further training and prediction. 
[Download the Brats2020 data from Kaggle](https://www.kaggle.com/awsaf49/brats20-dataset-training-validation) 
Used the 3D-Unet Model to train the Model with Loss function(dice_loss and folcal-loss), for Optimizer used Adam and for metrics used IoU score (greater than 0.5 is good for segmentaion).

## List of Pakages used are:
```
Numpy
nibabel
glob
matplotlib pyplot
splitfolders
sklearn.preprocessing - MinMaxScaler

tensorflow
keras
tensorflow.keras.utils - to_categorical
keras.models - Model, load_model
keras.layers  - Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout
tensorflow.keras.optimizers - Adam
keras.metrics - MeanIoU
segmentation_models_3D
```

Download external pakages used from: 
[Split Folders](https://pypi.org/project/split-folders) 
[Segmentation Model 3D](https://pypi.org/project/segmentation-models-3D)
