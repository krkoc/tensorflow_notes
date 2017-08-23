
__all__ = (
    'train',
)


import functools
import glob
import itertools
import multiprocessing
import random
import sys
import time

import cv2
import numpy
import tensorflow as tf



DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARS = LETTERS + DIGITS
print CHARS
def code_to_vec(p, code):
	def char_to_vec(c):
		y = numpy.zeros((len(CHARS),))
		y[CHARS.index(c)] = 1.0
		return y
	c = numpy.vstack([char_to_vec(c) for c in code])
	
	return numpy.concatenate([[1. if p else 0], c.flatten()])
	#return c
ctv=code_to_vec(1==1,"ABC4567")


print (ctv.shape)
print ctv[1]
#y_ = tf.placeholder(tf.float32, [None, 7 * len(CHARS) + 1])
#print(y_)
y_ = tf.placeholder(tf.float32, [None, 7 * len(CHARS) + 1])
print y_.shape

g=tf.reshape(y_[:,1:], [-1, len(CHARS)] )
print g.shape	