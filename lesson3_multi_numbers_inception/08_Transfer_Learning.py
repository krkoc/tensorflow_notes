
# coding: utf-8

# # TensorFlow Tutorial #08
# # Transfer Learning
# 
# by [Magnus Erik Hvass Pedersen](http://www.hvass-labs.org/)
# / [GitHub](https://github.com/Hvass-Labs/TensorFlow-Tutorials) / [Videos on YouTube](https://www.youtube.com/playlist?list=PL9Hr9sNUjfsmEu1ZniY0XpHSzl5uihcXZ)

# ## Introduction
# 
# We saw in the previous Tutorial #07 how to use the pre-trained Inception model for classifying images. Unfortunately the Inception model seemed unable to classify images of people. The reason was the data-set used for training the Inception model, which had some confusing text-labels for classes.
# 
# The Inception model is actually quite capable of extracting useful information from an image. So we can instead train the Inception model using another data-set. But it takes several weeks using a very powerful and expensive computer to fully train the Inception model on a new data-set.
# 
# We can instead re-use the pre-trained Inception model and merely replace the layer that does the final classification. This is called Transfer Learning.
# 
# This tutorial builds on the previous tutorials so you should be familiar with Tutorial #07 on the Inception model, as well as earlier tutorials on how to build and train Neural Networks in TensorFlow. A part of the source-code for this tutorial is located in the `inception.py` file.

# ## Flowchart

# The following chart shows how the data flows when using the Inception model for Transfer Learning. First we input and process an image with the Inception model. Just prior to the final classification layer of the Inception model, we save the so-called Transfer Values to a cache-file.
# 
# The reason for using a cache-file is that it takes a long time to process an image with the Inception model. My laptop computer with a Quad-Core 2 GHz CPU can process about 3 images per second using the Inception model. If each image is processed more than once then we can save a lot of time by caching the transfer-values.
# 
# The transfer-values are also sometimes called bottleneck-values, but that is a confusing term so it is not used here.
# 
# When all the images in the new data-set have been processed through the Inception model and the resulting transfer-values saved to a cache file, then we can use those transfer-values as the input to another neural network. We will then train the second neural network using the classes from the new data-set, so the network learns how to classify images based on the transfer-values from the Inception model.
# 
# In this way, the Inception model is used to extract useful information from the images and another neural network is then used for the actual classification.

# In[1]:


from IPython.display import Image, display
Image('images/08_transfer_learning_flowchart.png')


# ## Imports

# In[2]:


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os
import loadimages as li

# Functions and classes for loading and using the Inception model.
import inception

# We use Pretty Tensor to define the new classifier.
import prettytensor as pt
import cv2

# This was developed using Python 3.5.2 (Anaconda) and TensorFlow version:

# In[3]:


tf.__version__


# PrettyTensor version:

# In[4]:


pt.__version__


# ## Load Data for CIFAR-10 

# In[5]:


#import cifar10

#from cifar10 import num_classes

#cifar10.maybe_download_and_extract()

# compare classes
#class_names = cifar10.load_class_names()
#my_class_names = ["0","1","2","3","4","5","6","7","8","9"]
num_classes=10
print num_classes
#print ("class name type: ",type(my_class_names[0]))
#print("hvass classes ",class_names)
#print("my classes ",my_class_names)

#raw_input("press key to continue...")

# Load the training-set. This returns the images, the class-numbers as integers, and the class-numbers as One-Hot encoded arrays called labels.
# In[10]:


#images_train, cls_train, labels_train = cifar10.load_training_data()
my_labels_train_cls, my_images_train = li.load_label_image_list("/home/peter/tensorflow_scripts/triset_train/tags.csv", 0, 30000)
#truncate label for testing with first number onls



my_labels_train_onehot=[]
for x in my_labels_train_cls:
    my_labels_train_onehot.append(li.getonehot(x))
# Load the test-set.
#print("onehot : ",my_labels_train_onehot)
my_labels_train_onehot=np.array(my_labels_train_onehot)
print("onehot : ",my_labels_train_onehot.shape)
# In[11]:
print(my_labels_train_onehot    )
#raw_input("...")

#images_test, cls_test, labels_test = cifar10.load_test_data()

# The CIFAR-10 data-set has now been loaded and consists of 60,000 images and associated labels (i.e. classifications of the images). The data-set is split into 2 mutually exclusive sub-sets, the training-set and the test-set.

# In[12]:
    

print("Size of:")
print("- Training-set:\t\t{}".format(len(my_images_train    )))
#print("- Test-set:\t\t{}".format(len(images_test)))


# ### Helper-function for plotting images

# Function used to plot at most 9 images in a 3x3 grid, and writing the true and predicted classes below each image.

# In[13]:


def plot_images(images, cls_true, cls_pred=None, smooth=True):

    assert len(images) == len(cls_true)

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):
        # There may be less than 9 images, ensure it doesn't crash.
        if i < len(images):
            # Plot image.
            ax.imshow(images[i],
                      interpolation=interpolation)

            # Name of the true class.
            #cls_true_name = my_class_names[cls_true[i]]
            cls_true_name=str(cls_true[i])
            #cls_true_name="x"
            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                # Name of the predicted class.
                cls_pred_name = class_names[cls_pred[i]]

                xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# ### Plot a few images to see if data is correct

# In[14]:


# Get the first images from the test-set.
#images = images_test[0:9]
my_images=my_images_train[0:9]
#print ("hvass image shape", images[0].shape)
print ("my image shape", my_images[0].shape)
#raw_input("press..")
# Get the true classes for those images.
#cls_true = cls_test[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=my_images, cls_true=my_labels_train_cls[0:9], smooth=False)


inception.maybe_download()


# ## Load the Inception Model

# Load the Inception model so it is ready for classifying images.
# 
# Note the deprecation warning, which might cause the program to fail in the future.

# In[17]:


model = inception.Inception()


# ## Calculate Transfer-Values

# Import a helper-function for caching the transfer-values of the Inception model.

# In[18]:


from inception import transfer_values_cache


# Set the file-paths for the caches of the training-set and test-set.

# In[19]:


file_path_cache_train = "/home/peter/tensorflow_notes/train.pkl"
#file_path_cache_test = os.path.join(cifar10.data_path, 'inception_cifar10_test.pkl')


# In[20]:


print("Processing Inception transfer-values for training-images ...")

# Scale images because Inception needs pixels to be between 0 and 255,
# while the CIFAR-10 functions return pixels between 0.0 and 1.0
#images_scaled = images_train * 255.0

# If transfer-values have already been calculated then reload them,
# otherwise calculate them and save them to a cache-file.
my_transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,images=my_images_train, model=model)



print(my_transfer_values_train.shape)


def plot_transfer_values(i):
    print("Input image:")
    
    # Plot the i'th image from the test-set.
    plt.imshow(my_images_train[i], interpolation='nearest')
    plt.show()

    print("Transfer-values for the image using Inception model:")
    
    # Transform the transfer-values into an image.
    img = my_transfer_values_train[i]
    img = img.reshape((32, 64))

    # Plot the image for the transfer-values.
    plt.imshow(img, interpolation='nearest', cmap='Reds')
    plt.show()






x = tf.placeholder(tf.float32, shape=[None, 2048], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, 3,10], name='y_true')

#y_true_cls = tf.argmax(y_true, dimension=1)

y_true_cls = tf.argmax(tf.transpose(y_true)[0])*100+tf.argmax(tf.transpose(y_true)[1])*10+tf.argmax(tf.transpose(y_true)[2])

weights1 = tf.Variable(tf.zeros([2048, 1024]))
biases1 = tf.Variable(tf.zeros([1024]))
prelog = tf.matmul(x, weights1) + biases1
y_pred1 = tf.nn.softmax(prelog)


weights1f = tf.Variable(tf.zeros([1024, num_classes]))   
biases1f = tf.Variable(tf.zeros([num_classes]))

weights2f = tf.Variable(tf.zeros([1024, num_classes]))   
biases2f = tf.Variable(tf.zeros([num_classes]))

weights3f = tf.Variable(tf.zeros([1024, num_classes]))   
biases3f = tf.Variable(tf.zeros([num_classes]))

logits1 = tf.matmul(y_pred1, weights1f) + biases1f
logits2 = tf.matmul(y_pred1, weights2f) + biases2f
logits3 = tf.matmul(y_pred1, weights3f) + biases3f
logits= tf.stack ([logits1,logits2,logits3],axis=1)

y_pred = tf.nn.softmax(logits)
#y_pred=logits
#y_pred_cls = tf.argmax(y_pred, dimension=1)
y_pred_cls = tf.argmax(tf.transpose(y_pred)[0])*100+tf.argmax(tf.transpose(y_pred)[1])*10+tf.argmax(tf.transpose(y_pred)[2])


cross_entropy1 = tf.nn.softmax_cross_entropy_with_logits(logits=logits[0], labels=y_true[0])
cross_entropy2 = tf.nn.softmax_cross_entropy_with_logits(logits=logits[1], labels=y_true[1])
cross_entropy3 = tf.nn.softmax_cross_entropy_with_logits(logits=logits[2], labels=y_true[2])
cross_entropy = tf.reduce_sum([cross_entropy1,cross_entropy2,cross_entropy3])


loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=5).minimize(loss)

y_pred_cls = tf.argmax(tf.transpose(y_pred)[0])*100+tf.argmax(tf.transpose(y_pred)[1])*10+tf.argmax(tf.transpose(y_pred)[2])

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)

#optimizer = tf.train.AdamOptimizer(learning_rate=1e-6).minimize(loss, global_step)



session = tf.Session()

session.run(tf.global_variables_initializer())
#session.run(tf.initialize_all_variables() )

train_batch_size=10
def get_batch(start_position, train_batch_size):
    # Number of images (transfer-values) in the training-set.

    x_batch = my_transfer_values_train[start_position: start_position+train_batch_size]
    y_batch = my_labels_train_onehot[start_position: start_position+train_batch_size]

    return x_batch, y_batch


# ### Helper-function to perform optimization

# This function performs a number of optimization iterations so as to gradually improve the variables of the neural network. In each iteration, a new batch of data is selected from the training-set and then TensorFlow executes the optimizer using those training samples.  The progress is printed every 100 iterations.
# In[56]:


def optimize(num_iterations):
    # Start-time used for printing time-usage below.
    start_time = time.time()
    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images (transfer-values) and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = get_batch(i*train_batch_size,train_batch_size)
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        # We also want to retrieve the global_step counter.
        session.run(optimizer, feed_dict=feed_dict_train)

         # Print status to screen every 100 iterations (and last).
        if (i % 100 == 0) or (i == num_iterations - 1):
            # Calculate the accuracy on the training-batch.
            batch_acc = session.run(accuracy,  feed_dict=feed_dict_train)
            # Print status.
            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.3%}"
            print(msg.format(i, batch_acc))

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


# ## Helper-Functions for Showing Results

# ### Helper-function to plot example errors

# Function for plotting examples of images from the test-set that have been mis-classified.

# In[57]:


def plot_example_errors(cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = images_test[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = cls_test[incorrect]

    n = min(9, len(images))
    
    # Plot the first n images.
    plot_images(images=images[0:n],
                cls_true=cls_true[0:n],
                cls_pred=cls_pred[0:n])


# ### Helper-function to plot confusion matrix

# In[58]:


# Import a function from sklearn to calculate the confusion-matrix.
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(num_classes):
        # Append the class-name to each line.
        class_name = "({}) {}".format(i, class_names[i])
        print(cm[i, :], class_name)

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(num_classes)]
    print("".join(class_numbers))


# ### Helper-functions for calculating classifications
# 
# This function calculates the predicted classes of images and also returns a boolean array whether the classification of each image is correct.
# 
# The calculation is done in batches because it might use too much RAM otherwise. If your computer crashes then you can try and lower the batch-size.

# In[59]:


# Split the data-set in batches of this size to limit RAM usage.
batch_size = 256

def predict_cls(transfer_values, labels, cls_true):
    # Number of images.
    num_images = len(transfer_values)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {x: transfer_values[i:j],
                     y_true: labels[i:j]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j
        
    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred


# Calculate the predicted class for the test-set.

# In[60]:


def predict_cls_test():
    return predict_cls(transfer_values = transfer_values_test,
                       labels = labels_test,
                       cls_true = cls_test)


# ### Helper-functions for calculating the classification accuracy
# 
# This function calculates the classification accuracy given a boolean array whether each image was correctly classified. E.g. `classification_accuracy([True, True, False, False, False]) = 2/5 = 0.4`. The function also returns the number of correct classifications.

# In[61]:


def classification_accuracy(correct):
    # When averaging a boolean array, False means 0 and True means 1.
    # So we are calculating: number of True / len(correct) which is
    # the same as the classification accuracy.

    # Return the classification accuracy
    # and the number of correct classifications.
    return correct.mean(), correct.sum()


# ### Helper-function for showing the classification accuracy

# Function for printing the classification accuracy on the test-set.
# 
# It takes a while to compute the classification for all the images in the test-set, that's why the results are re-used by calling the above functions directly from this function, so the classifications don't have to be recalculated by each function.

# In[62]:


def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # For all the images in the test-set,
    # calculate the predicted classes and whether they are correct.
    correct, cls_pred = predict_cls_test()
    
    # Classification accuracy and the number of correct classifications.
    acc, num_correct = classification_accuracy(correct)
    
    # Number of images being classified.
    num_images = len(correct)

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)



optimize(num_iterations=3000)



img = cv2.imread("/home/peter/tensorflow_scripts/triset_train/461_21127.png")
image= np.asarray(img)


res=session.run(y_pred, feed_dict={x: my_transfer_values_train[2:30]})
for i in res:
    print np.amax(i)
print res    
print ("number :", my_labels_train_cls[2:30])

# In[65]:


    