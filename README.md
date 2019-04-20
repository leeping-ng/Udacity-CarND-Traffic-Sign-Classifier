# Udacity-CarND-Traffic-Sign-Classifier
A deep learning project that uses a CNN to classify traffic signs, for the Udacity Self-Driving Car Engineer Nanodegree.

The objective of this project is to build a Convolutional Neural Network (CNN) that can distinguish between the 43 different road signs of the German Traffic Signs Dataset.

Some sample images of the dataset are shown below.

<img src='https://github.com/leeping-ng/Udacity-CarND-Traffic-Sign-Classifier/blob/master/writeup_images/original_images.JPG' width=1000>

### Step 1: Analyse the Dataset
The first step is to explore the data provided.

- Number of training examples = 34799
- Number of validation examples = 4410
- Number of testing examples = 12630
- Image data shape = (32, 32, 3)
- Number of classes = 43

I plotted the histogram below to see the number of samples for each class.
<img src='https://github.com/leeping-ng/Udacity-CarND-Traffic-Sign-Classifier/blob/master/writeup_images/original_dataset.JPG' width=1000>

The training dataset looks very unbalanced, with some classes having about 180 examples only, while others having more than 2,000 examples. If trained using this dataset, there might be biasedness or overfitting to the classes with large number of examples.

### Step 2: Augment the Dataset
To prevent overfitting, we can build a jittered dataset by synthetically adding transformed versions of the original training and validation sets.

From Pierre Sermanet and Yann LeCun's paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks", it was stated that "ConvNets architectures have built-in invariance to small translations, scaling and rotations. When a dataset does not naturally contain those deformations, adding them synthetically will yield more robust learning to potential deformations in the test set. Other realistic perturbations would probably also increase robustness such as other affine transformations, brightness, contrast and blur."

Random small translations, scaling and rotations were introduced to augment the dataset, as shown below.
<img src='https://github.com/leeping-ng/Udacity-CarND-Traffic-Sign-Classifier/blob/master/writeup_images/augmented_images.JPG'>

The number of training examples was increased from 34799 to 46480 by adding augmented images, and the minimum number of samples per class was increased to 800. The number of validation examples was increased from 4410 to 5630 by adding augmented images, and the minimum number of samples per class was increased to 100. 
<img src='https://github.com/leeping-ng/Udacity-CarND-Traffic-Sign-Classifier/blob/master/writeup_images/augmented_dataset.JPG'>

### Step 3: Pre-Process the Dataset
From Pierre Sermanet and Yann LeCun's paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks", using grayscale instead of colour images increased the performance of the neural network from 98.97% to 99.17%. Therefore, grayscaling was used to pre-process the data.

Also, the image data was normalized so that the data has mean zero and equal variance. (pixel - 128)/ 128 was used as a quick way to approximately normalize the data.

<img src='https://github.com/leeping-ng/Udacity-CarND-Traffic-Sign-Classifier/blob/master/writeup_images/pre-processed_images.JPG'>

### Step 4: Model Architecture
The LeNet-5 convolutional neural network architecture was used, with 2 convolutional layers and 3 fully connected layers. A slight modification to the original LeNet-5 architecture which I did was to add dropout to the fully connected layers, which significantly improved performance.

<img src='https://github.com/leeping-ng/Udacity-CarND-Traffic-Sign-Classifier/blob/master/writeup_images/lenet5_architecture.JPG'>

I used the Adam optimizer, which is similar to stochastic gradient descent, to minimize the loss function. I kept to an epoch of 30 usually, varying batch size between 64 and 128, and learning rate between 0.0007 to 0.0013. I tweaked these hyperparameters to improve the validation accuracy.

### Step 5: Validation and Testing Results
After tuning hyperparameters, the model was able to achieve **94.5-95.5%** validation accuracy on the validation dataset, exceeding the project requirement of 93.0%.

When tested on the test dataset, the model had a test accuracy of **94.2%**, which was only slightly lower than the validation accuracy. This shows that the trained model does not have much overfitting, and that the measures of introducing augmented data and dropout were effective in removing biasedness to the training and validation datasets.

### Step 6: Testing on New Images
5 new German traffic sign images were taken from the web, and tested on the model. The model was able to achieve 100% accuracy on the 5 images. The softmax probabilities showed that the model was very certain when making predictions.

<img src='https://github.com/leeping-ng/Udacity-CarND-Traffic-Sign-Classifier/blob/master/writeup_images/new_images.JPG'>
