#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.



**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1_1]: ./img/visualization.jpg "Visualization"
[image1_2]: ./img/visualization_0.jpg "Visualization0"
[image1_3]: ./img/visualization_1.jpg "Visualization1"
[image1_4]: ./img/visualization_2.jpg "Visualization2"
[image1_5]: ./img/visualization_3.jpg "Visualization2"




[image2]: ./img/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"

[image4]: ./data/p1.jpg "Traffic Sign 1"
[image5]: ./data/p2.jpg "Traffic Sign 2"
[image6]: ./data/p3.jpg "Traffic Sign 3"
[image7]: ./data/p4.jpg "Traffic Sign 4"
[image8]: ./data/p5.jpg "Traffic Sign 5"
[image9]: ./img/top5.jpg "Top 5 predict result"



## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/xh2012/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributed between classes.

![visualization of the class][image1_1]

Here is an random overview of the images order by class
![alt text][image1_2]
and here comes more images...
![alt text][image1_3]




###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

First i relize the examples numbers per class is not balanced. so i **resampling** the classes with less examples. So each class have 2010 examples.
![alt text][image1_4]

then, I decided to convert the images to grayscale because after convert into grayscale, the model trained more faster....

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because it training more faster and seems to have a higher Validating Accuracy......(and so the deep learning is totally a black magic....)

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

At first I use train.p and test.p, I use **train_test_split** function from sklearn.model_selection to split data into training/validating/test with the ratio 6/2/2.(Use the train_test_split twice, first split it into 6/4 , the split the second part into 5/5)

And now I just used the "train.p" "valid.p" "test.p" files udacity provided, so i skipped this step. 
...


My final training set had 86430 number of images. My validation set and test set had 4410 and 12630 number of images.



####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description											| 
|:---------------------:|:----------------------------------------------------------:| 
| Input         		| 32x32x1 gray_normized image								|
| Convolution 5x5     	| 1x1 stride,6 filters, valid padding, outputs 28x28x6 		|
| RELU					|															|
| Max pooling 2x2     	| 2x2 stride, valid padding, outputs 14x14x6 				|
| Convolution 5x5	    | 2x2 stride,16 filters, valid padding, outputs 10x10x16 	|
| RELU					|       													|
| Max pooling 2x2		| 2x2 stride, valid padding, outputs 10x10x16 				|
| Flatten				| outputs 400												|
| Fully Connected		| inputs 400	outputs 300									|
| RELU					|															|
| DROP_OUT				| drop_out_prob = 0.5										|
| Fully Connected		| inputs 300	outputs 200									|
| RELU					|															|
| DROP_OUT				| 															|
| Fully Connected		| inputs 200	outputs 43									|


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used Adam Optimizer with 200 epochs with batch_size 128 the learning rate is 0.001

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 1.00
* validation set accuracy of 0.97
* test set accuracy of 0.95

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?


I choose LeNet as my first architecture,Because it was simple.First i change the finally output nerouns number to fit this problem.I do not have much experience in find tuning a neural network and there were so many parameter and hyper-parameter. 
* At First, I try to let the training accuracy converge to 1.000 first. then the validation accuracy..
* so i did something such as adding nerouns in the fully connected layers and it works. 
* Chanage the RGB image to Gray seems to speed the converge process makes it possible to get 90% in less than 5 epochs. 
* Normalize the image to -1 to 1 seems make the training error get 90% at the first epochs.

* Then the validation accuracy.
* adding a drop_out layer, improved the validation accuracy from 89% to 92%
* adding another drop_out layer, improved the validation accuracy from 92% to 94%
* balance the data, makes the validation accuracy from 94% to 97% 





If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?



###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The third image might be difficult to classify because the image was not at the center of the crop image, and there were another traffic sign behind it, it may have some difficulty to predict on this image.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).


Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (120km/h) | Speed limit (120km/h)  						| 
| Double curve     		| Yield									|
| Roundabout mandatory	| Speed limit (30km/h)							|
| Bicycles crossing	    | Beware of Ice/Snow				 				|
| Stop					| Stop      									|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 60%. This compares favorably to the accuracy on the test set of 95%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


![alt text][image9]

the model seems to be certain at every image it predict.And then it gets wrong in the 3rd image, It mistake the "Roudabout Mandatory" as a "Speed Limit(30km/h)", but "Roudabout Mandatory" is in the 4th place .....

