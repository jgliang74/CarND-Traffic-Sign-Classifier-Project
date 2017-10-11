# Traffic Sign Recognition

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./report_images/raw_trainingset_images.png "Sample training images"
[image2]: ./report_images/class_distribution.png "Class distribution"
[image3]: ./report_images/normalized_trainingset_images.png "Image normalization"
[image4]: ./extra_images/1.png "Traffic Sign 1"
[image5]: ./extra_images/2.png "Traffic Sign 2"
[image6]: ./extra_images/3.png "Traffic Sign 3"
[image7]: ./extra_images/4.png "Traffic Sign 4"
[image8]: ./extra_images/5.png "Traffic Sign 5"

#### Data Set Summary & Exploration

###### 1. Provide a basic summary of the data set.
I used the pandas library to calculate summary statistics of the traffic signs data set:
* The size of training set is  34799
* The size of the validation set is  12630
* The size of test set is  4410
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

###### 2. Include an exploratory visualization of the dataset.

Here are exploratory visualization of the data set. Random selected images from the training data set are displayed with corresponding label ID on the left label. The bottom bar charts are showing how frequently each label occurs in training set, validation set and test set respectively.

![aaa][image1]
![bbb][image2]

#### Design and Test a Model Architecture

###### 1. Pre-process the Data Set

Only two basic pre-process steps, image normalization and RGB-to-grayscale are implemented to facilitate subseuqent model trainning. Example images after pre-processing are shown as followings. Because the main focus of this project is to go through the whole process of deep learning network development. I didn't spend too much time on trying additional pre-process steps such as data augumentation with rotation, flipping, etc which could  futher improving model training accuracy.

![ccc][image3]

###### 2. Final model architecture 
My final model consisted of the following layers.
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image 						| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x64 	|
| RELU                  |                                               |
| Max pooling           | 2x2 stride, outputs 5x5x64                    |
| Fully connected		| Input = 1600. Output = 120      				|
| RELU                  |                                               |
| Drop out				| keep prob = 0.8       						|
| Fully connected 		| Input = 120. Output = 84						|
| RELU					|												|
| Drop out              | keep prob = 0.8                               |
| Fully connected       | Input = 120. Output = 43                      |
| Regularizer/softmax   |                                               |   

It is based on LeNet model introudced in class and make the following enhancements:
a). Added drop out after two fully conneted layers
b). L2 regularization to penalize larger weights.
c). Increased number of parameters (first CNN's output went up to 28x28x32), this increases model training time on my local machine, but using GPU from AWS helped to speed things up.

###### 3. Model training
To train the model, I use adam optimizer with learning rate at 0.001. Weights and bias for each layers are generated using normal distribution with mu = 0 and sigma = 0.1. Training batch size is set to 128. In order to avoid overfitting, the EPOCHS is set to 20

###### 4. Training iteration and accuracy improvement 
Using the initial LeNet model without data pre-processing, the validation accuracy is around 85%-88%. In order to further improve the accuracy, the first thing I did is to pre-process the training data with RGB to grayscale followed by the image normalization. This immediately increased the accuracy level slightly above 90% (around 91%). 
The next improvement made was to add dropouts after two fully connected layers. It works pretty well by increasing accuracy level close to 94%-95%. Keep Prob is a tuning knob for it. I settled down on the value of 0.8 (meaning 20% information will be cleared for each epoch)
The last step is to increase the convolution layer's depth. I used 32 and 64 for the depth of first convolution and second convolution layer repectively. This allowed more featuremaps generated based on the training data.

My final model results were:
* training set accuracy of  99.6%
* validation set accuracy of 97.3%
* test set accuracy of 95.2%

#### Test a Model on New Images

###### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image4] ![alt text][image8]

The third image (class id 27) might be difficult to classify because it is one the least occured signs (210 times) in the training set. Plus due to the poor image resolution, there are certain level of similarity between it and other signs, such as "Road Work" (class id 25). Both of then have a human shape in the center of image.

###### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Slippery Road      	| Slippery Road   								| 
| Wild animals crossing | Wild animals crossing							|
| Pedestrians			| Pedestrians									|
| Dangerous curve to the right| Dangerous curve to the right			|
| Stop		        	| Stop              							|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95.2%. This may be attributed to the supreme image quality.

###### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Slippery Road sign (probability of 0.99), and the image indeed is a Slippery Road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .993         			| Slippery Road   								| 
| .0057     			| Dangerous curve to the right 					|
| .0008					| Right-of-way at the next intersection			|
| .00009	     		| Children crossing				 				|
| .000007			    | Beware of ice/snow      						|

For the second image, the model almost hundred percent sure this is a Wild animal crossing sign. The distant second possiblity is only 7.1e08 (virutally impossible)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| Wild animals crossing   						| 
| 7.1e-08      			| Road work			                        	|
| 5.6e-10				| Double curve		                        	|
| 3.1e-11	     		| Speed limit (50km/h)				 			|
| 1.5e-12			    | Right-of-way at the next intersection 		|

image 3, even though very small occurences of this sign presented in the training set. The model recognize this sign without any problem. It is almost 99% sure for its pick.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .988         			| Pedestrians           						| 
| 0.00416      			| Road work			                        	|
| 0.00399				| General caution		                    	|
| 0.00347	     		| Road narrows on the right			 			|
| 0.000088			    | Traffic signals 	                         	|

image 4, looks like a piece of cake.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Dangerous curve to the right   				| 
| 3.3e-13      			| End of no passing			                 	|
| 7.8e-15				| Slippery road		                        	|
| 3.8e-16	     		| No passing			        	 			|
| 8.6e-20			    | Children crossing	                        	|

image 5, again no chanllenges at all for this one.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Stop      				            		| 
| 1.9e-8      			| Go straight or right	                     	|
| 2.6e-10				| Turn left ahead	                        	|
| 4.0e-11	     		| Speed limit (80km/h)			     			|
| 3.2e-12			    | Ahead only                            		|

#### Visualizing the Neural Network
For understanding the outputs of a network's weights, the feature maps of layer1 and layer2 convolution steps were plotted as shown in the last section of genreated Traffic_sign_classifier.html. 
The stimuli image (input impage) is a stop sign and the network were fully trained. The layer1 convolution's activations clearly picked up the edges and brighter pixels outline painted word "STOP". The layer2 convolution's feature maps seems to be more abstracted than layer1's feature maps with unrelated information (or noises) being squeezed out of the maps. Most maps are kind of hard for human to recognize, but I can vaguely tell highly simplifiy letter "OP" on one feature map (featuremap 62 for example)





