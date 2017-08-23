import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
#from tensorflow.examples.tutorials.mnist import input_data
#data = input_data.read_data_sets("data/MNIST/", one_hot=True)
#data.test.cls = np.array([label.argmax() for label in data.test.labels])
import os
import cv2
np.set_printoptions(threshold='nan')
# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)  

# Number of classes, one class for each of 10 digits.
num_classes = 10

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)  
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):

        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
       
        xlabel = "True: {0}".format(cls_true[i])
       
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
# In[39]:

def getonehot(value):
    precursor=np.array([0,0,0,0,0,0,0,0,0,0,])
    precursor[value]=1
    return precursor

# Get the first images from real files
filelist=[]
images=[]
print("reading data...")
i =0
for subdir, dirs, files in os.walk('/home/peter/tensorflow_scripts/nminst'):
    for file in files:
        if file.endswith (".png") and i < 10000 :
          if i %1000==0:
            print(i)  
          i=i+1
          truenum=int(file[0])  
         
          filelist.append(truenum)
          im = cv2.imread("/home/peter/tensorflow_scripts/nminst/"+file)
          image= np.asarray(im)
          

          image=  1-np.transpose(image)[2].flatten()/256.    
          images.append(image)

npfiles=np.array(filelist) 
npimages=np.array(images)
onehotfilelist=[]
for x in npfiles:
    onehotfilelist.append(getonehot(x))
nponehotfilelist=np.array(onehotfilelist)
#print nponehotfilelist.shape
#print nponehotfilelist
#print images[4]

# Get the true classes for those images.
cls_true = npfiles[0:9]
print type(cls_true)
print npimages[1].shape
print cls_true.shape
# Plot the images and labels using our helper-function above.

def nextBatch(iterations,batch_size):
    return ( npimages[iterations*batch_size:iterations*batch_size+batch_size],nponehotfilelist[iterations*batch_size:iterations*batch_size+batch_size])

# In[51]:


x = tf.placeholder(tf.float32, [None, img_size_flat])



y_true = tf.placeholder(tf.float32, [None, num_classes])

y_true_cls = tf.placeholder(tf.int64, [None])


weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))

biases = tf.Variable(tf.zeros([num_classes]))
logits = tf.matmul(x, weights) + biases
y_pred = tf.nn.softmax(logits)


y_pred_cls = tf.argmax(y_pred, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())
batch_size = 100
feed_dict_test = {x: npimages,
                  y_true: npfiles,
                  y_true_cls: nponehotfilelist}
x1,y1 =nextBatch(1,100)
print x1.shape
print y1.shape



def optimize(num_iterations):
    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = nextBatch(i,batch_size)
              # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        # Note that the placeholder for y_true_cls is not set
        # because it is not used during training.
        feed_dict_train = {x: x_batch, y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.


# In[53]:

   


optimize(100) #perform 100 optimisations

print "exiting optimization"

img = cv2.imread("/home/peter/tensorflow_scripts/nminst/1_715.png")
image= np.asarray(img)
image=  np.transpose(image)[1].flatten()    
newvar=1-np.array(image/256.)
#print newvar
newvar = np.reshape(newvar, (-1, 784))
#print(newvar)

res=session.run(y_pred, feed_dict={x: newvar})
print res

# In[57]:


#data.test.cls[0]


# In[ ]:




