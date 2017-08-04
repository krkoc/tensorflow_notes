# tensorflow_notes
These are my first documented steps into TensorFlow. The python examples borrow (very) heavily from Hvass-Labs tutorials. Please bear in mind that my contributions are just quick and dirty ways to find out how things are done in TensorFlow. Unfortunately I have no time or interest to make nice polished tutorials in the near future. If you have a hard time following the code, please refer to the Hvass-Labs tutorials first (https://github.com/Hvass-Labs/TensorFlow-Tutorials).

lesson1_simple_CNN:
The thing that bothered me was a myriad of tutorials on how to train and evaluate a TF model with the MNIST dataset, but not a single one that would show how to use ones own images for that. 

The hello world example in this lesson uses a custom dataset of 28x28 images with numbers instead of the MNIST dataset. It uses the script “opencvexperiments.py” to produce a desired number of training images as single digit numbers of the decimal system. The images have a fixed font but vary in size, position, rotation and stretch. These images can be fed into the simple Y=aX+b model (or any model with the same input interface) in TensorFlow.



License (MIT)

These tutorials and source-code are published under the MIT License which allows very broad use for both academic and commercial purposes.

A few of the images used for demonstration purposes may be under copyright. These images are included under the "fair usage" laws.

You are very welcome to modify these tutorials and use them in your own projects. Please keep a link to the original repository.
