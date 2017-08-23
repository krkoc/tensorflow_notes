import csv
import numpy as np
import cv2
#create dictionary from file

def load_file_dict(file, start, batch_size):
	#file= CSV file with [value, image] for rows
	#start = starting point of read
	#batch_size= num of elements to read. start + batch_size must be less than num of rows!  
	fhandle=open(file,'r')
	lines=fhandle.readlines()
	nplines=np.array(lines)
	part=nplines[start:start+batch_size]
	return part

def load_label_image_list(file, start, batch_size):
	true_labels=[]
	image_file_names=[]
	images=[]
	dict=load_file_dict(file, start, batch_size)
	for x in dict:
		splitx=x.split(",",1)
		true_labels.insert(-1,int(splitx[0]))
		image_file_names.insert(-1,splitx[1])	
	for x in image_file_names:	
		im = cv2.imread(x[:-1]) #drop the final newline character
		image= np.asarray(im)
		images.append(image)
	return true_labels, images


def getonehot(value):
	precursor=([0,0,0,0,0,0,0,0,0,0])
	precursor[value]=1
	#precursor[20+	ones]=1
	#precursor[0,hundreds]=1
	#precursor[1,tens]=1
	#precursor[2,ones]=1
	return precursor


#print 	getonehot(243).shape
#a=load_label_image_list("/home/peter/tensorflow_scripts/triset_train/tags.csv",10,12)	

#print 