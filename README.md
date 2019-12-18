
# Convolutional Neural Network Hand Emoji Detector with Tensorflow Extended (TFX) 
Hand emoji recognition by using convolutional neural network implemented with Keras, Theano, and OpenCV

# Team Members 

We are a five-member team: 
- Ma Jia
- Yuanzhe Li
- Ying Liu
- Yuhua He
- Samuel Yang

# Technical Requirements:
- Python 3
- OpenCV 
- Keras 
- Tensorflow 
- Theano 
- Matplotlib
- tfx.component: SchemaGen, Trainer, Evaluator, Transform, ModelValidator, Pusher, Bulkinferrer
- Apache Airflow
- Package versions are specified in the requriements.txt file in the repository.

# Repository file contents
- **trackhandemoji.py** : This is our main function. This file contains all the code for user interface options and OpenCV code to capture camera contents. This file internally calls interfaces to HandEmojiCNN.py.
- **HandEmojiCNN.py** : This file holds all the CNN specific code to create CNN model, load the weight file (if model is pretrained), train the model using image samples present in **./imgfolder_b**, visualize the feature maps at different layers of NN (of pretrained model) for a given input image present in **./imgs** folder.
- **imgfolder_b** : This folder contains all the 4015 hand emoji images to train the model. You need to unzip the file inside.
- **_pretrained_weights_MacOS.hdf5_** : This is pretrained weight file on MacOS. Due to its large size (150 MB), its hosted seperately on this google driver link - https://drive.google.com/file/d/1j7K96Dkatz6q6zr5RsQv-t68B3ZOSfh0/view. You need to download this and save it to the application folder before running the app. 
- **_imgs_** - This folder has some sample images to visualize feature layer maps at different layers along with App Demo images for ReadMe display. We have multiple layer images pasted below.

# Colab links 
for trackhandemoji.ipynb and HandEmojiCNN.ipynb : (SJSU accounts have view access to below links) 
https://colab.research.google.com/drive/1_pPYctqgU4mS8Y33uYAuzh9jvhYrl1Fe (HandEmojiCNN.ipynb)
https://colab.research.google.com/drive/1MjzqiI5bgT2Aw67Ety7d6L6TQMw3Ee1f (Trackhandemoji.ipynb) This colab shows visulization of various image layers, and tensor board)

# Implementation from local terminal 
**On Mac only** (We tested this application on multiple mac laptops.)
```bash
With Theano as backend
$ KERAS_BACKEND=theano pythonw trackhandemoji.py 
```
We are setting KERAS_BACKEND to change backend to Theano.

# Input and Output

- We use the 4015 images to train the model. 
- Users could use pre-trained weights without training the model. In this case, input is the hand emoji images captured from the computer camera. 
- Output: application will predict the hand emoji's names. We have 5 names: OK, Peace, Stop, Punch, Nothing.

# App Demo

**local terminal**

![local terminal](https://github.com/wise-monk123/CMPE297DeepLearningProject/blob/master/imgs/Demo1.png)


**camera capture**

![camera capture](https://github.com/wise-monk123/CMPE297DeepLearningProject/blob/master/imgs/Demo2.png)


**image processing**

![image processing](https://github.com/wise-monk123/CMPE297DeepLearningProject/blob/master/imgs/Demo3.png)


**prediction result**

![Option1 opens up 3 windows](https://github.com/wise-monk123/CMPE297DeepLearningProject/blob/master/imgs/demo5.png)


# Features
This application comes with CNN model to recognize up to 5 pretrained gestures:
- OK
- PEACE
- STOP
- PUNCH
- NOTHING (ie when none of the above gestures are input)

This application provides following functionalities:
- Prediction : this allows the app to guess the user's gesture against pretrained gestures. App can dump the prediction data to the console terminal or to a json file directly which can be used to plot real time prediction bar chart
- Training : this allows user to retrain the CNN model. User can change the model architecture or add/remove new gestures. This app has inbuilt options to allow users to create new image samples of user defined gestures if required.
- Visualization : this allows the user to see feature maps of different neural network layers for a given input gesture image. As illustrated in the layer images section below, each image will be saved with neural network layer name, such as activation, max pooling etc.  


# Hand Emoji Input
We are using OpenCV for capturing user's hand gestures. In order to simplify the prediction, we are doing post processing on the captured images to highlight the contours & edges, such as applying binary threshold, blurring, gray scaling.

There are two modes of image capturing:
- Binary Mode : we first convert the image to grayscale, then apply a gaussian blur effect with adaptive threshold filter. This mode is useful when you have an empty background like a wall, whiteboard etc.
- SkinMask Mode : we first convert the input image to HSV and put range on the H,S,V values based on skin color range. Then apply errosion followed by dilation. Then gaussian blur to smoothen out the noises. Using this output as a mask on original input to mask out everything other than skin colored things. Finally I have grayscaled it. This mode is useful when there is good amount of light and you dont have empty background.

**Binary Mode**
```python
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),2)   
th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
```

![OK gesture in Binary mode](https://github.com/wise-monk123/CMPE297DeepLearningProject/blob/master/imgs/iiiok1.png)


**SkindMask Mode**
```python
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
#Apply skin color range
mask = cv2.inRange(hsv, low_range, upper_range)

mask = cv2.erode(mask, skinkernel, iterations = 1)
mask = cv2.dilate(mask, skinkernel, iterations = 1)

#blur
mask = cv2.GaussianBlur(mask, (15,15), 1)
#cv2.imshow("Blur", mask)

#bitwise and mask original frame
res = cv2.bitwise_and(roi, roi, mask = mask)
# color to grayscale
res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
```
![OK gesture in SkinMask mode](https://github.com/wise-monk123/CMPE297DeepLearningProject/blob/master/imgs/iiok44.png)


# CNN Model Architecture

```python
model = Sequential()
model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                    padding='valid',
                    input_shape=(img_channels, img_rows, img_cols)))
convout1 = Activation('relu')
model.add(convout1)
model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
```

This model has following 12 layers -
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 32, 198, 198)      320       
_________________________________________________________________
activation_1 (Activation)    (None, 32, 198, 198)      0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 32, 196, 196)      9248      
_________________________________________________________________
activation_2 (Activation)    (None, 32, 196, 196)      0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 32, 98, 98)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 32, 98, 98)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 307328)            0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               39338112  
_________________________________________________________________
activation_3 (Activation)    (None, 128)               0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 5)                 645       
_________________________________________________________________
activation_4 (Activation)    (None, 5)                 0         
=================================================================
```
Total params: 39,348,325.0
Trainable params: 39,348,325.0

# Training
We have trained the model for 15 epochs,and used loss & accuracy graphs for measurement. The results can be viewd in Colab links as well.

![Training Accuracy Vs Validation Accuracy](https://github.com/wise-monk123/CMPE297DeepLearningProject/blob/master/ori_4015imgs_acc.png)

![Training Loss Vs Validation Loss](https://github.com/wise-monk123/CMPE297DeepLearningProject/blob/master/ori_4015imgs_loss.png)


# Visualization

After launching the main script, choose option 3 for visualizing different or all layer for a given image (currently it takes images from ./imgs, so change it accordingly)
```
What would you like to do ?
    1- Use pretrained model for gesture recognition & layer visualization
    2- Train the model (you will require image samples for training under .\imgfolder)
    3- Visualize feature maps of different layers of trained model
    3
Will load default weight file
Image number 7
Enter which layer to visualize -1
(4015, 40000)
Press any key
samples_per_class -  803
Total layers - 12
Dumping filter data of layer1 - Activation
Dumping filter data of layer2 - Conv2D
Dumping filter data of layer3 - Activation
Dumping filter data of layer4 - MaxPooling2D
Dumping filter data of layer5 - Dropout
Can't dump data of this layer6- Flatten
Can't dump data of this layer7- Dense
Can't dump data of this layer8- Activation
Can't dump data of this layer9- Dropout
Can't dump data of this layer10- Dense
Can't dump data of this layer11- Activation
Press any key to continue
```

To understand how its done in Keras, check visualizeLayer() in gestureCNN.py
```python
layer = model.layers[layerIndex]

get_activations = K.function([model.layers[0].input, K.learning_phase()], [layer.output,])
activations = get_activations([input_image, 0])[0]
output_image = activations
```

Layer 1 visualization for OK gesture
![Layer 1 visualization for OK gesture](https://github.com/wise-monk123/CMPE297DeepLearningProject/blob/master/img_1_layer1_Activation.png)

Layer 4 visualization for PUNCH gesture
![Layer 4 visualization for PUNCH gesture](https://github.com/wise-monk123/CMPE297DeepLearningProject/blob/master/img_4_layer4_MaxPooling2D.png)

Layer 2 visualization for STOP gesture
![Layer 2 visualization for STOP gesture](https://github.com/wise-monk123/CMPE297DeepLearningProject/blob/master/img_7_layer2_Conv2D.png)

Layer 1 visualization for NOTHING gesture
![Layer 1 visualization for NOTHING gesture](https://github.com/wise-monk123/CMPE297DeepLearningProject/blob/master/img_0_layer1_Activation.png)

# TPU, CPU, GPU Comparision
![TPU](https://github.com/wise-monk123/CMPE297DeepLearningProject/blob/master/imgs/CPU_TPU_GPU)


# Tensor Board
![Tensorboard](https://github.com/wise-monk123/CMPE297DeepLearningProject/blob/master/imgs/Tensorboard.png)


# Tensorflow Extended (TFX)

Open the browser to 127.0.0.1:8080. Here is the Graph view: 

![Tensorboard](https://github.com/wise-monk123/CMPE297DeepLearningProject/blob/master/imgs/graphview.png)
