# Image Classifier, a CNN to classify images
CNN designed to classify images of cats and dogs (or just about anything and everything) 

It is a 3 layer deep network, designed to solve the following problem [Dogs vs. Cats | Kaggle](https://www.kaggle.com/c/dogs-vs-cats)
but can be used to classify any given set of labeled images.

### reqirements :
* tensorflow
* numpy
* cv2

### prepare the dataset using
`python prepare_data.py`

scales down all the images to the same resolution as described in the hyperparameters

### train 
`python run_train.py`

