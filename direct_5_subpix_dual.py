# Hidden 2 domains no constrained
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, argparse, glob

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function


# Misc. libraries
from six.moves import map, zip, range
from natsort import natsorted 

# Array and image processing toolboxes
import numpy as np 
import skimage
import skimage.io
import skimage.transform
import skimage.segmentation
import malis

# Tensorpack toolbox
import tensorpack.tfutils.symbolic_functions as symbf

from tensorpack import *
from tensorpack.utils.viz import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.utils.utils import get_rng
from tensorpack.utils.argtools import memoized
from tensorpack import (TowerTrainer,
						ModelDescBase, DataFlow, StagingInput)
from tensorpack.tfutils.tower import TowerContext, TowerFuncWrapper
from tensorpack.graph_builder import DataParallelBuilder, LeastLoadedDeviceSetter

from tensorpack.tfutils.summary import add_moving_summary, add_tensor_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.varreplace import freeze_variables

# Tensorflow 
import tensorflow as tf
from GAN import GANTrainer, GANModelDesc, SeparateGANTrainer, MultiGPUGANTrainer
from tensorlayer.cost import binary_cross_entropy, absolute_difference_error, dice_coe
from sklearn.metrics.cluster import adjusted_rand_score

###############################################################################
SHAPE = 256
BATCH = 1
TEST_BATCH = 100
EPOCH_SIZE = 100
NB_FILTERS = 16	  # channel size

DIMX  = 512
DIMY  = 512
DIMZ  = 5
DIMC  = 1

MAX_LABEL = 320
###############################################################################
import tensorflow as tf
import numpy as np
from malis import nodelist_like, malis_loss_weights



def seg_to_aff_op(seg, nhood=tf.constant(malis.mknhood3d(1)), name=None):
	# Squeeze the segmentation to 3D
	seg = tf.squeeze(seg)
	# Define the numpy function to transform segmentation to affinity graph
	np_func = lambda seg, nhood: malis.seg_to_affgraph (seg, nhood).astype(np.float32)
	# Convert the numpy function to tensorflow function
	tf_func = tf.py_func(np_func, [seg, nhood], [tf.float32], name=name)
	# Reshape the result, notice that layout format from malis is 3, dimx, dimy, dimx
	ret = tf.reshape(tf_func[0], [3, seg.shape[0], seg.shape[1], seg.shape[2]])
	# Transpose the result so that the dimension 3 go to the last channel
	ret = tf.transpose(ret, [1, 2, 3, 0])
	# print ret.get_shape().as_list()
	return ret

def aff_to_seg_op(aff, nhood=tf.constant(malis.mknhood3d(1)), threshold=tf.constant(np.array([0.5])), name=None):
	# Define the numpy function to transform affinity to segmentation
	def np_func (aff, nhood, threshold):
		aff = np.transpose(aff, [3, 0, 1, 2]) # zyx3 to 3zyx
		ret = malis.connected_components_affgraph((aff > threshold[0]).astype(np.int32), nhood)[0].astype(np.int32) 
		ret = skimage.measure.label(ret).astype(np.int32)
		return ret
	# print aff.get_shape().as_list()
	# Convert numpy function to tensorflow function
	tf_func = tf.py_func(np_func, [aff, nhood, threshold], [tf.int32], name=name)
	ret = tf.reshape(tf_func[0], [aff.shape[0], aff.shape[1], aff.shape[2]])
	ret = tf.expand_dims(ret, axis=-1)
	# print ret.get_shape().as_list()
	return ret

def tf_rand_score (x1, x2):
	def np_func (x1, x2):
		ret = np.mean(1.0 - adjusted_rand_score (x1.flatten (), x2.flatten ()))
		return ret
	tf_func = tf.py_func(np_func, [x1,  x2], [tf.float64])
	ret = tf_func[0]
	ret = tf.cast(ret, tf.float32)
	return ret

# def tf_rand_score (x1, x2):
# 	return np.mean(1.0 - adjusted_rand_score (x1.flatten (), x2.flatten ()))

def toMaxLabels(label, factor=MAX_LABEL):
	result = tf.cast(label, tf.float32)
	status = tf.equal(result, -1.0*tf.ones_like(result))
	result = tf.where(status, tf.zeros_like(result), result, name='removedBackground') # From -1 to 0
	result = result * factor # From 0~1 to 0~MAXLABEL
	result = tf.round(result)
	return tf.cast(result, tf.int32)


def toRangeTanh(label, factor=MAX_LABEL):
	label  = tf.cast(label, tf.float32)
	result = label / factor # From 0~MAXLABEL to 0~1
	status = tf.equal(result, 0.0*tf.zeros_like(result))
	result = tf.where(status, -1.0*tf.ones_like(result), result, name='addedBackground') # From -1 to 0
	return tf.cast(result, tf.float32)

###############################################################################
def magnitute_central_difference(image, name=None):
	from tensorflow.python.framework import ops
	from tensorflow.python.ops import math_ops
	with ops.name_scope(name, 'magnitute_central_difference'):
		ndims = image.get_shape().ndims
		Gx = tf.zeros_like(image)
		Gy = tf.zeros_like(image)

		if ndims == 3:
			pass
		elif ndims == 4:
			# The input is a batch of image with shape:
			# [batch, height, width, channels].

			# Calculate the difference of neighboring pixel-values.
			# The image are shifted one pixel along the height and width by slicing.
			padded_img1 = tf.pad(image, paddings=[[0,0], [1,1], [0,0], [0,0]], mode="REFLECT")
			padded_img2 = tf.pad(image, paddings=[[0,0], [0,0], [1,1], [0,0]], mode="REFLECT")
			# padded_img3 = tf.pad(image, paddings=[[1,1], [0,0], [0,0], [0,0]], mode="REFLECT")
			
			Gx = 0.5*(padded_img1[:,:-2,:,:] - padded_img1[:,2:,:,:])
			Gy = 0.5*(padded_img2[:,:,:-2,:] - padded_img2[:,:,2:,:])
			# Gz = 0.5*(padded_img3[:-2,:,:,:] - padded_img3[2:,:,:,:])
			# grad = tf.sqrt(tf.add_n([tf.pow(Gx,2),tf.pow(Gy,2),tf.pow(Gz,2)]))
			# return grad
		else:
			raise ValueError('\'image\' must be either 3 or 4-dimensional.')

		grad = tf.sqrt(tf.add(tf.square(Gx),tf.square(Gy))) # okay
		return grad

		loss_img = tf_2tanh(loss_img)
		return loss_val, loss_img
###############################################################################
def INReLU(x, name=None):
	x = InstanceNorm('inorm', x)
	return tf.nn.relu(x, name=name)


def INLReLU(x, name=None):
	x = InstanceNorm('inorm', x)
	return tf.nn.leaky_relu(x, name=name)
	
def BNLReLU(x, name=None):
	x = BatchNorm('bn', x)
	return tf.nn.leaky_relu(x, name=name)

###############################################################################
# Utility function for scaling 
def tf_2tanh(x, maxVal = 255.0, name='ToRangeTanh'):
	with tf.variable_scope(name):
		return (x / maxVal - 0.5) * 2.0
###############################################################################
def tf_2imag(x, maxVal = 255.0, name='ToRangeImag'):
	with tf.variable_scope(name):
		return (x / 2.0 + 0.5) * maxVal

# Utility function for scaling 
def np_2tanh(x, maxVal = 255.0, name='ToRangeTanh'):
	return (x / maxVal - 0.5) * 2.0
###############################################################################
def np_2imag(x, maxVal = 255.0, name='ToRangeImag'):
	return (x / 2.0 + 0.5) * maxVal

###############################################################################
# FusionNet
@layer_register(log_shape=True)
def residual(x, chan, first=False):
	with argscope([Conv2D], nl=INLReLU, stride=1, kernel_shape=3):
		input = x
		return (LinearWrap(x)
				.Conv2D('conv1', chan, padding='SAME', dilation_rate=1)
				.Conv2D('conv2', chan, padding='SAME', dilation_rate=2)
				.Conv2D('conv4', chan, padding='SAME', dilation_rate=4)				
				.Conv2D('conv5', chan, padding='SAME', dilation_rate=8)
				.Conv2D('conv0', chan, padding='SAME', nl=tf.identity)
				.InstanceNorm('inorm')()) + input

###############################################################################
@layer_register(log_shape=True)
def Subpix2D(inputs, chan, scale=2, stride=1):
	with argscope([Conv2D], nl=INLReLU, stride=stride, kernel_shape=3):
		results = Conv2D('conv0', inputs, chan* scale**2, padding='SAME')
		old_shape = inputs.get_shape().as_list()
		# results = tf.reshape(results, [-1, chan, old_shape[2]*scale, old_shape[3]*scale])
		# results = tf.reshape(results, [-1, old_shape[1]*scale, old_shape[2]*scale, chan])
		if scale>1:
			results = tf.depth_to_space(results, scale, name='depth2space', data_format='NHWC')
		return results

###############################################################################
@layer_register(log_shape=True)
def residual_enc(x, chan, first=False):
	with argscope([Conv2D, Deconv2D], nl=INLReLU, stride=1, kernel_shape=3):
		x = (LinearWrap(x)
			# .Dropout('drop', 0.75)
			.Conv2D('conv_i', chan, stride=2) 
			.residual('res_', chan, first=True)
			.Conv2D('conv_o', chan, stride=1) 
			())
		return x

###############################################################################
@layer_register(log_shape=True)
def residual_dec(x, chan, first=False):
	with argscope([Conv2D, Deconv2D], nl=INLReLU, stride=1, kernel_shape=3):
				
		x = (LinearWrap(x)
			.Subpix2D('deconv_i', chan, scale=1) 
			.residual('res2_', chan, first=True)
			.Subpix2D('deconv_o', chan, scale=2) 
			# .Dropout('drop', 0.75)
			())
		return x

###############################################################################
@auto_reuse_variable_scope
def arch_generator(img, last_dim=1):
	assert img is not None
	with argscope([Conv2D, Deconv2D], nl=INLReLU, kernel_shape=3, stride=2, padding='SAME'):
		e0 = residual_enc('e0', img, NB_FILTERS*1)
		e1 = residual_enc('e1',  e0, NB_FILTERS*2)
		e2 = residual_enc('e2',  e1, NB_FILTERS*4)

		e3 = residual_enc('e3',  e2, NB_FILTERS*8)
		# e3 = Dropout('dr', e3, 0.5)

		d3 = residual_dec('d3',    e3, NB_FILTERS*4)
		d2 = residual_dec('d2', d3+e2, NB_FILTERS*2)
		d1 = residual_dec('d1', d2+e1, NB_FILTERS*1)
		d0 = residual_dec('d0', d1+e0, NB_FILTERS*1) 
		dd =  (LinearWrap(d0)
				.Conv2D('convlast', last_dim, kernel_shape=3, stride=1, padding='SAME', nl=tf.tanh, use_bias=True) ())
		return dd, d0

@auto_reuse_variable_scope
def arch_discriminator(img):
	assert img is not None
	with argscope([Conv2D, Deconv2D], nl=INLReLU, kernel_shape=3, stride=2, padding='SAME'):
		img = Conv2D('conv0', img, NB_FILTERS, nl=tf.nn.leaky_relu)
		e0 = residual_enc('e0', img, NB_FILTERS*1)
		# e0 = Dropout('dr', e0, 0.5)
		e1 = residual_enc('e1',  e0, NB_FILTERS*2)
		e2 = residual_enc('e2',  e1, NB_FILTERS*4)

		e3 = residual_enc('e3',  e2, NB_FILTERS*8)

		ret = Conv2D('convlast', e3, 1, stride=1, padding='SAME', nl=tf.identity, use_bias=True)
		return ret


###############################################################################
class ImageDataFlow(RNGDataFlow):
	def __init__(self, imageDir, labelDir, size, dtype='float32', isTrain=False, isValid=False, isTest=False):
		self.dtype      = dtype
		self.imageDir   = imageDir
		self.labelDir   = labelDir
		self._size      = size
		self.isTrain    = isTrain
		self.isValid    = isValid

	def size(self):
		return self._size

	def reset_state(self):
		self.rng = get_rng(self)

	def get_data(self, shuffle=True):
		#
		# Read and store into pairs of images and labels
		#
		images = glob.glob(self.imageDir + '/*.tif')
		labels = glob.glob(self.labelDir + '/*.tif')

		if self._size==None:
			self._size = len(images)

		from natsort import natsorted
		images = natsorted(images)
		labels = natsorted(labels)


		#
		# Pick the image over size 
		#
		for k in range(self._size):
			#
			# Pick randomly a tuple of training instance
			#
			rand_index = np.random.randint(0, len(images))
			image_p = skimage.io.imread(images[rand_index])
			label_p = skimage.io.imread(labels[rand_index])
			membr_p = label_p.copy()


			#
			# Pick randomly a tuple of training instance
			#
			rand_image = np.random.randint(0, len(images))
			rand_membr = np.random.randint(0, len(images))
			rand_label = np.random.randint(0, len(images))


			# image_u = skimage.io.imread(images[rand_image])
			# membr_u = skimage.io.imread(labels[rand_membr])
			# label_u = skimage.io.imread(labels[rand_label])
			image_u = image_p.copy() #skimage.io.imread(images[rand_index])
			membr_u = membr_p.copy() #skimage.io.imread(labels[rand_index])
			label_u = label_p.copy() #skimage.io.imread(labels[rand_index])


			# Cut 1 or 3 slices along z, by define DIMZ, the same for paired, randomly for unpaired
			dimz, dimy, dimx = image_u.shape

			seed = np.random.randint(0, 20152015)
			seed_image = np.random.randint(0, 2015)
			seed_membr = np.random.randint(0, 2015)
			seed_label = np.random.randint(0, 2015)
			np.random.seed(seed)

			# The same for pair
			randz = np.random.randint(0, dimz-DIMZ+1)
			randy = np.random.randint(0, dimy-DIMY+1)
			randx = np.random.randint(0, dimx-DIMX+1)
			headx = np.random.randint(0, 2)
			heady = np.random.randint(0, 2)
			# image_p = image_p[randz:randz+DIMZ,randy:randy+DIMY,randx:randx+DIMX]
			# membr_p = membr_p[randz:randz+DIMZ,randy:randy+DIMY,randx:randx+DIMX]
			# label_p = label_p[randz:randz+DIMZ,randy:randy+DIMY,randx:randx+DIMX]
			image_p = image_p[randz:randz+DIMZ,heady::2,headx::2]
			membr_p = membr_p[randz:randz+DIMZ,heady::2,headx::2]
			label_p = label_p[randz:randz+DIMZ,heady::2,headx::2]

			# Randomly for unpaired for pair
			randz = np.random.randint(0, dimz-DIMZ+1)
			randy = np.random.randint(0, dimy-DIMY+1)
			randx = np.random.randint(0, dimx-DIMX+1)
			# image_u = image_u[randz:randz+DIMZ,randy:randy+DIMY,randx:randx+DIMX]
			image_u = image_u[randz:randz+DIMZ,heady::2,headx::2]
			randz = np.random.randint(0, dimz-DIMZ+1)
			randy = np.random.randint(0, dimy-DIMY+1)
			randx = np.random.randint(0, dimx-DIMX+1)
			# membr_u = membr_u[randz:randz+DIMZ,randy:randy+DIMY,randx:randx+DIMX]
			membr_u = membr_u[randz:randz+DIMZ,heady::2,headx::2]
			randz = np.random.randint(0, dimz-DIMZ+1)
			randy = np.random.randint(0, dimy-DIMY+1)
			randx = np.random.randint(0, dimx-DIMX+1)
			# label_u = label_u[randz:randz+DIMZ,randy:randy+DIMY,randx:randx+DIMX]
			label_u = label_u[randz:randz+DIMZ,heady::2,headx::2]



			
			
			



			if self.isTrain:
				# Augment the pair image for same seed
				image_p = self.random_flip(image_p, seed=seed)        
				image_p = self.random_reverse(image_p, seed=seed)
				image_p = self.random_square_rotate(image_p, seed=seed)           
				image_p = self.random_elastic(image_p, seed=seed)

				membr_p = self.random_flip(membr_p, seed=seed)        
				membr_p = self.random_reverse(membr_p, seed=seed)
				membr_p = self.random_square_rotate(membr_p, seed=seed)   
				membr_p = self.random_elastic(membr_p, seed=seed)

				label_p = self.random_flip(label_p, seed=seed)        
				label_p = self.random_reverse(label_p, seed=seed)
				label_p = self.random_square_rotate(label_p, seed=seed)   
				label_p = self.random_elastic(label_p, seed=seed)
				
				# Augment the unpair image for different seed seed
				image_u = self.random_flip(image_u, seed=seed_image)        
				image_u = self.random_reverse(image_u, seed=seed_image)
				image_u = self.random_square_rotate(image_u, seed=seed_image)           
				image_u = self.random_elastic(image_u, seed=seed_image)

				membr_u = self.random_flip(membr_u, seed=seed_membr)        
				membr_u = self.random_reverse(membr_u, seed=seed_membr)
				membr_u = self.random_square_rotate(membr_u, seed=seed_membr)   
				membr_u = self.random_elastic(membr_u, seed=seed_membr)

				label_u = self.random_flip(label_u, seed=seed_label)        
				label_u = self.random_reverse(label_u, seed=seed_label)
				label_u = self.random_square_rotate(label_u, seed=seed_label)   
				label_u = self.random_elastic(label_u, seed=seed_label)


			# Calculate membrane
			def membrane(label):
				membr = np.zeros_like(label)
				for z in range(membr.shape[0]):
					membr[z,...] = 1-skimage.segmentation.find_boundaries(np.squeeze(label[z,...]), mode='thick') #, mode='inner'
				membr = 255*membr
				membr[label==0] = 0 
				return membr

			membr_p = membrane(membr_p.copy())
			membr_u = membrane(membr_u.copy())

			# label_p = label_p[label_p>0] + 1.0
			# label_u = label_u[label_u>0] + 1.0

			# Calculate linear label
			label_p, nb_labels_p = skimage.measure.label(label_p.copy(), return_num=True)
			label_u, nb_labels_u = skimage.measure.label(label_u.copy(), return_num=True)

			label_p = label_p.astype(np.float32)
			label_u = label_u.astype(np.float32)

			label_p = label_p / MAX_LABEL
			label_u = label_u / MAX_LABEL

			label_p[membr_p==0] = -1.0
			label_u[membr_u==0] = -1.0

			label_p = np_2imag(label_p, maxVal=255.0)
			label_u = np_2imag(label_u, maxVal=255.0)

			#Expand dim to make single channel
			image_p = np.expand_dims(image_p, axis=-1)
			membr_p = np.expand_dims(membr_p, axis=-1)
			label_p = np.expand_dims(label_p, axis=-1)

			image_u = np.expand_dims(image_u, axis=-1)
			membr_u = np.expand_dims(membr_u, axis=-1)
			label_u = np.expand_dims(label_u, axis=-1)

			yield [image_p.astype(np.float32), 
				   membr_p.astype(np.float32), 
				   label_p.astype(np.float32), 
				   image_u.astype(np.float32), 
				   membr_u.astype(np.float32), 
				   label_u.astype(np.float32),
				   ] 

	def random_flip(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		if seed:
			np.random.seed(seed)
		random_flip = np.random.randint(1,5)
		if random_flip==1:
			flipped = image[...,::1,::-1]
			image = flipped
		elif random_flip==2:
			flipped = image[...,::-1,::1]
			image = flipped
		elif random_flip==3:
			flipped = image[...,::-1,::-1]
			image = flipped
		elif random_flip==4:
			flipped = image
			image = flipped
		return image

	def random_reverse(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		if seed:
			np.random.seed(seed)
		random_reverse = np.random.randint(1,3)
		if random_reverse==1:
			reverse = image[::1,...]
		elif random_reverse==2:
			reverse = image[::-1,...]
		image = reverse
		return image

	def random_square_rotate(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		if seed:
			np.random.seed(seed)        
		random_rotatedeg = 90*np.random.randint(0,4)
		rotated = image.copy()
		from scipy.ndimage.interpolation import rotate
		if image.ndim==2:
			rotated = rotate(image, random_rotatedeg, axes=(0,1))
		elif image.ndim==3:
			rotated = rotate(image, random_rotatedeg, axes=(1,2))
		image = rotated
		return image
				
	def random_elastic(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		old_shape = image.shape

		if image.ndim==2:
			image = np.expand_dims(image, axis=0) # Make 3D
		new_shape = image.shape
		dimx, dimy = new_shape[1], new_shape[2]
		size = np.random.randint(4,16) #4,32
		ampl = np.random.randint(2, 5) #4,8
		du = np.random.uniform(-ampl, ampl, size=(size, size)).astype(np.float32)
		dv = np.random.uniform(-ampl, ampl, size=(size, size)).astype(np.float32)
		# Done distort at boundary
		du[ 0,:] = 0
		du[-1,:] = 0
		du[:, 0] = 0
		du[:,-1] = 0
		dv[ 0,:] = 0
		dv[-1,:] = 0
		dv[:, 0] = 0
		dv[:,-1] = 0
		import cv2
		from scipy.ndimage.interpolation    import map_coordinates
		# Interpolate du
		DU = cv2.resize(du, (new_shape[1], new_shape[2])) 
		DV = cv2.resize(dv, (new_shape[1], new_shape[2])) 
		X, Y = np.meshgrid(np.arange(new_shape[1]), np.arange(new_shape[2]))
		indices = np.reshape(Y+DV, (-1, 1)), np.reshape(X+DU, (-1, 1))
		
		warped = image.copy()
		for z in range(new_shape[0]): #Loop over the channel
			# print z
			imageZ = np.squeeze(image[z,...])
			flowZ  = map_coordinates(imageZ, indices, order=0).astype(np.float32)

			warpedZ = flowZ.reshape(image[z,...].shape)
			warped[z,...] = warpedZ     
		warped = np.reshape(warped, old_shape)
		return warped


class Model(GANModelDesc):
	#FusionNet
	@auto_reuse_variable_scope
	def generator(self, img, last_dim=1):
		assert img is not None
		return arch_generator(img, last_dim=last_dim)
		# return arch_fusionnet(img)

	@auto_reuse_variable_scope
	def discriminator(self, img):
		assert img is not None
		return arch_discriminator(img)


	def _get_inputs(self):
		return [
			InputDesc(tf.float32, (DIMZ, DIMY, DIMX, 1), 'image_p'),
			InputDesc(tf.float32, (DIMZ, DIMY, DIMX, 1), 'membr_p'),
			InputDesc(tf.float32, (DIMZ, DIMY, DIMX, 1), 'label_p'),
			InputDesc(tf.float32, (DIMZ, DIMY, DIMX, 1), 'image_u'),
			InputDesc(tf.float32, (DIMZ, DIMY, DIMX, 1), 'membr_u'),
			InputDesc(tf.float32, (DIMZ, DIMY, DIMX, 1), 'label_u'),
			]
	def build_losses(self, vecpos, vecneg, name="WGAN_loss"):
		with tf.name_scope(name=name):
			# the Wasserstein-GAN losses
			d_loss = tf.reduce_mean(vecneg - vecpos, name='d_loss')
			g_loss = tf.negative(tf.reduce_mean(vecneg), name='g_loss')
			# add_moving_summary(self.d_loss, self.g_loss)
			return g_loss, d_loss

	def _build_graph(self, inputs):
		G = tf.get_default_graph() # For round
		tf.local_variables_initializer()
		tf.global_variables_initializer()
		pi, pm, pl, ui, um, ul = inputs
		pi = tf_2tanh(pi)
		pm = tf_2tanh(pm)
		pl = tf_2tanh(pl)
		ui = tf_2tanh(ui)
		um = tf_2tanh(um)
		ul = tf_2tanh(ul)


		



		with argscope([Conv2D, Deconv2D, FullyConnected],
					  W_init=tf.truncated_normal_initializer(stddev=0.02),
					  use_bias=False), \
				argscope(BatchNorm, gamma_init=tf.random_uniform_initializer()), \
				argscope([Conv2D, Deconv2D, BatchNorm], data_format='NHWC'), \
				argscope([Conv2D], dilation_rate=1):

			

			with tf.variable_scope('gen'):
				
				with tf.variable_scope('affnt'):
					pia, feat_ia  = self.generator(pi, last_dim=3)
				with tf.variable_scope('label'):
					pil, feat_il  = self.generator(pi, last_dim=1)
			


			# 
			with G.gradient_override_map({"round": "Identity"}):
				with tf.variable_scope('fix'):
					# Round
					pil  = toMaxLabels(pil, factor=MAX_LABEL) #0 MAX
					pil  = toRangeTanh(pil, factor=MAX_LABEL) # -1 1
					
					pa   = seg_to_aff_op(toMaxLabels(pl, factor=MAX_LABEL),   name='pa') # Calculate the affinity 	#0, 1
					pila = seg_to_aff_op(toMaxLabels(pil, factor=MAX_LABEL),  name='pila') # Calculate the affinity 	#0, 1

					pial = aff_to_seg_op(tf_2imag(pia, maxVal=1.0), name='pial') # Calculate the segmentation
					pial = toRangeTanh(pial, factor=MAX_LABEL) # -1, 1

					pil_ = (pial + pil) / 2.0 # Return the result
					pia_ = (pila + pia) / 2.0

			with tf.variable_scope('discrim'):
				print pial
				print pil
				dis_real = self.discriminator(tf.concat([pi, pl, pa], axis=-1))
				dis_fake = self.discriminator(tf.concat([pi, pil, pia], axis=-1))

			with tf.name_scope('GAN_loss'):
				G_loss, D_loss = self.build_losses(dis_real, dis_fake, name='gan_loss')

			with tf.name_scope('rand_loss'):
				rand_il  = tf.reduce_mean(tf_rand_score(pl, pil_), name='rand_loss')
				# rand_il  = tf.reduce_mean(tf.cast(tf.py_func (tf_rand_score, [pl, pil_], tf.float64), tf.float32), name='rand_loss')
			with tf.name_scope('discrim_loss'):
				def regDLF(y_true, y_pred, alpha=1, beta=1, gamma=0.01, delta_v=0.5, delta_d=1.5, name='loss_discrim'):
					def tf_norm(inputs, axis=1, epsilon=1e-7,  name='safe_norm'):
						squared_norm 	= tf.reduce_sum(tf.square(inputs), axis=axis, keep_dims=True)
						safe_norm 		= tf.sqrt(squared_norm+epsilon)
						return tf.identity(safe_norm, name=name)
					###


					lins = tf.linspace(0.0, DIMZ*DIMY*DIMX, DIMZ*DIMY*DIMX)
					lins = tf.cast(lins, tf.int32)
					# lins = lins / tf.reduce_max(lins) * 255
					# lins = tf_2tanh(lins)
					# lins = tf.reshape(lins, tf.shape(y_true), name='lins_3d')
					# print lins
					lins_z = tf.div(lins,(DIMY*DIMX))
					lins_y = tf.div(tf.mod(lins,(DIMY*DIMX)), DIMY)
					lins_x = tf.mod(tf.mod(lins,(DIMY*DIMX)), DIMY)

					lins   = tf.cast(lins  , tf.float32)
					lins_z = tf.cast(lins_z, tf.float32)
					lins_y = tf.cast(lins_y, tf.float32)
					lins_x = tf.cast(lins_x, tf.float32)

					lins   = lins 	/ tf.reduce_max(lins) * 255
					lins_z = lins_z / tf.reduce_max(lins_z) * 255
					lins_y = lins_y / tf.reduce_max(lins_y) * 255
					lins_x = lins_x / tf.reduce_max(lins_x) * 255

					lins   = tf_2tanh(lins)
					lins_z = tf_2tanh(lins_z)
					lins_y = tf_2tanh(lins_y)
					lins_x = tf_2tanh(lins_x)

					lins   = tf.reshape(lins,   tf.shape(y_true), name='lins')
					lins_z = tf.reshape(lins_z, tf.shape(y_true), name='lins_z')
					lins_y = tf.reshape(lins_y, tf.shape(y_true), name='lins_y')
					lins_x = tf.reshape(lins_x, tf.shape(y_true), name='lins_x')

					y_true = tf.reshape(y_true, [DIMZ*DIMY*DIMX])
					y_pred = tf.concat([y_pred, lins, lins_z, lins_y, lins_x], axis=-1)

					nDim = tf.shape(y_pred)[-1]
					X = tf.reshape(y_pred, [DIMZ*DIMY*DIMX, nDim])
					uniqueLabels, uniqueInd = tf.unique(y_true)

					numUnique = tf.size(uniqueLabels) # Get the number of connected component

					Sigma = tf.unsorted_segment_sum(X, uniqueInd, numUnique)
					# ones_Sigma = tf.ones((tf.shape(X)[0], 1))
					ones_Sigma = tf.ones_like(X)
					ones_Sigma = tf.unsorted_segment_sum(ones_Sigma, uniqueInd, numUnique)
					mu = tf.divide(Sigma, ones_Sigma)

					Lreg = tf.reduce_mean(tf.norm(mu, axis=1, ord=1))

					T = tf.norm(tf.subtract(tf.gather(mu, uniqueInd), X), axis = 1, ord=1)
					T = tf.divide(T, Lreg)
					T = tf.subtract(T, delta_v)
					T = tf.clip_by_value(T, 0, T)
					T = tf.square(T)

					ones_Sigma = tf.ones_like(uniqueInd, dtype=tf.float32)
					ones_Sigma = tf.unsorted_segment_sum(ones_Sigma, uniqueInd, numUnique)
					clusterSigma = tf.unsorted_segment_sum(T, uniqueInd, numUnique)
					clusterSigma = tf.divide(clusterSigma, ones_Sigma)

					# Lvar = tf.reduce_mean(clusterSigma, axis=0)
					Lvar = tf.reduce_mean(clusterSigma)

					mu_interleaved_rep = tf.tile(mu, [numUnique, 1])
					mu_band_rep = tf.tile(mu, [1, numUnique])
					mu_band_rep = tf.reshape(mu_band_rep, (numUnique*numUnique, nDim))

					mu_diff = tf.subtract(mu_band_rep, mu_interleaved_rep)
							# Remove zero vector
							# intermediate_tensor = reduce_sum(tf.abs(x), 1)
							# zero_vector = tf.zeros(shape=(1,1), dtype=tf.float32)
							# bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
							# omit_zeros = tf.boolean_mask(x, bool_mask)
					intermediate_tensor = tf.reduce_sum(tf.abs(mu_diff), 1)
					zero_vector = tf.zeros(shape=(1,1), dtype=tf.float32)
					bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
					omit_zeros = tf.boolean_mask(mu_diff, bool_mask)
					mu_diff = tf.expand_dims(omit_zeros, axis=1)
					print mu_diff
					mu_diff = tf.norm(mu_diff, ord=1)
							# squared_norm = tf.reduce_sum(tf.square(s), axis=axis,keep_dims=True)
							# safe_norm = tf.sqrt(squared_norm + epsilon)
							# squared_norm = tf.reduce_sum(tf.square(omit_zeros), axis=-1,keep_dims=True)
							# safe_norm = tf.sqrt(squared_norm + 1e-6)
							# mu_diff = safe_norm

					mu_diff = tf.divide(mu_diff, Lreg)

					mu_diff = tf.subtract(2*delta_d, mu_diff)
					mu_diff = tf.clip_by_value(mu_diff, 0, mu_diff)
					mu_diff = tf.square(mu_diff)

					numUniqueF = tf.cast(numUnique, tf.float32)
					Ldist = tf.reduce_mean(mu_diff)        

					# L = alpha * Lvar + beta * Ldist + gamma * Lreg
					# L = tf.reduce_mean(L, keep_dims=True)
					L = tf.reduce_sum([alpha*Lvar, beta*Ldist, gamma*Lreg], keep_dims=False)
					print L
					print Ldist
					print Lvar
					print Lreg
					return tf.identity(L,  name=name)
				discrim_il  = regDLF(toMaxLabels(pl, factor=MAX_LABEL), 
									 tf.concat([feat_il, feat_ia], axis=-1), name='discrim_il')
			with tf.name_scope('recon_loss'):		
				recon_il = tf.reduce_mean(tf.abs(pl - pil_), name='recon_il')
			with tf.name_scope('affnt_loss'):		
				# affnt_il = tf.reduce_mean(tf.abs(pa - pia_), name='affnt_il')
				affnt_il = tf.reduce_mean(tf.subtract(binary_cross_entropy(pa, pia_), 
								   dice_coe(pa, pia_, axis=[0,1,2,3], loss_type='jaccard')))
			with tf.name_scope('residual_loss'):		
				# residual_a = tf.reduce_mean(tf.abs(pia - pila), name='residual_a')
				# residual_l = tf.reduce_mean(tf.abs(pil - pial), name='residual_l')
				residual_a = tf.reduce_mean(tf.cast(tf.not_equal(pia, pila), tf.float32), name='residual_a')
				residual_l = tf.reduce_mean(tf.cast(tf.not_equal(pil, pial), tf.float32), name='residual_l')
				residual_il = tf.reduce_mean([residual_a, residual_l], name='residual_il')
			def label_imag(y_pred_L, name='label_imag'):
				mag_grad_L   = magnitute_central_difference(y_pred_L, name='mag_grad_L')
				cond = tf.greater(mag_grad_L, tf.zeros_like(mag_grad_L))
				thresholded_mag_grad_L = tf.where(cond, 
										   tf.ones_like(mag_grad_L), 
										   tf.zeros_like(mag_grad_L), 
										   name='thresholded_mag_grad_L')

				thresholded_mag_grad_L = tf_2tanh(thresholded_mag_grad_L, maxVal=1.0)
				return thresholded_mag_grad_L

			g_il =  label_imag(pil_, name='label_il')
			g_l  =  label_imag(pl,   name='label_l')

		self.g_loss = tf.reduce_sum([
								1*(G_loss), 
								10*(recon_il), 
								10*(residual_il), 
								1*(rand_il), 
								50*(discrim_il), 
								0.005*affnt_il, 		
								], name='G_loss_total')
		self.d_loss = tf.reduce_sum([
								D_loss
								], name='D_loss_total')
		wd_g = regularize_cost('gen/.*/W', 		l2_regularizer(1e-5), name='G_regularize')
		wd_d = regularize_cost('discrim/.*/W', 	l2_regularizer(1e-5), name='D_regularize')

		self.g_loss = tf.add(self.g_loss, wd_g, name='g_loss')
		self.d_loss = tf.add(self.d_loss, wd_d, name='d_loss')

	

		self.collect_variables()

		add_moving_summary(self.d_loss, self.g_loss)
		with tf.name_scope('summaries'):	
			add_tensor_summary(recon_il, 		types=['scalar'], name='recon_il')
			add_tensor_summary(rand_il, 		types=['scalar'], name='rand_il')
			add_tensor_summary(discrim_il, 		types=['scalar'], name='discrim_il')
			add_tensor_summary(affnt_il, 		types=['scalar'], name='affnt_il')
			add_tensor_summary(residual_il, 	types=['scalar'], name='residual_il')

		#Segmentation
		viz = tf.concat([tf.concat([pi, pl, g_l, g_il ], 2), 
						 tf.concat([tf.zeros_like(pi), pil, pial, pil_], 2), 
						 
						 ], 1)
		
		viz = tf_2imag(viz)
		viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
		tf.summary.image('colorized', viz, max_outputs=50)

		# Affinity
		vis = tf.concat([pa, pila, pia, pia_ ], 2)
		
		vis = tf_2imag(vis)
		vis = tf.cast(tf.clip_by_value(vis, 0, 255), tf.uint8, name='vis')
		tf.summary.image('affinities', vis, max_outputs=50)



	def _get_optimizer(self):
		lr = symbolic_functions.get_scalar_var('learning_rate', 2e-4, summary=True)
		return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)
###############################################################################
class VisualizeRunner(Callback):
	def _setup_graph(self):
		self.pred = self.trainer.get_predictor(
			['image_p', 'membr_p', 'label_p', 'image_u', 'membr_u', 'label_u'], ['viz'])

	def _before_train(self):
		global args
		self.test_ds = get_data(args.data, isTrain=False, isValid=False, isTest=True)

	def _trigger(self):
		for lst in self.test_ds.get_data():
			viz_test = self.pred(lst)
			viz_test = np.squeeze(np.array(viz_test))

			#print viz_test.shape

			self.trainer.monitors.put_image('viz_test', viz_test)
###############################################################################
def get_data(dataDir, isTrain=False, isValid=False, isTest=False):
	# Process the directories 
	if isTrain:
		num=500
		names = ['trainA', 'trainB']
	if isValid:
		num=1
		names = ['trainA', 'trainB']
	if isTest:
		num=1
		names = ['testA', 'testB']

	
	dset  = ImageDataFlow(os.path.join(dataDir, names[0]),
						  os.path.join(dataDir, names[1]),
						  num, 
						  isTrain=isTrain, 
						  isValid=isValid, 
						  isTest =isTest)
	dset.reset_state()
	return dset
###############################################################################
class ClipCallback(Callback):
	def _setup_graph(self):
		vars = tf.trainable_variables()
		ops = []
		for v in vars:
			n = v.op.name
			if not n.startswith('discrim/'):
				continue
			logger.info("Clip {}".format(n))
			ops.append(tf.assign(v, tf.clip_by_value(v, -0.01, 0.01)))
		self._op = tf.group(*ops, name='clip')

	def _trigger_step(self):
		self._op.run()
###############################################################################
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu',    help='comma seperated list of GPU(s) to use.')
	parser.add_argument('--data',   required=True, 
									help='Data directory, contain trainA/trainB/validA/validB')
	parser.add_argument('--load',   help='Load the model path')
	parser.add_argument('--sample', help='Run the deployment on an instance',
									action='store_true')

	args = parser.parse_args()
	# python Exp_FusionNet2D_-VectorField.py --gpu='0' --data='arranged/'

	
	train_ds = get_data(args.data, isTrain=True, isValid=False, isTest=False)
	# valid_ds = get_data(args.data, isTrain=False, isValid=True, isTest=False)
	# test_ds  = get_data(args.data, isTrain=False, isValid=False, isTest=True)

	# train_ds = PrintData(train_ds)
	# valid_ds = PrintData(valid_ds)
	# test_ds  = PrintData(test_ds)
	# Augmentation is here
	

	# data_set  = ConcatData([train_ds, valid_ds])
	data_set  = train_ds
	# data_set  = LocallyShuffleData(data_set, buffer_size=4)
	# data_set  = AugmentImageComponent(data_set, augmentors, (0)) # Only apply for the image


	data_set  = PrintData(data_set)
	data_set  = PrefetchDataZMQ(data_set, 8)
	data_set  = QueueInput(data_set)
	model 	  = Model()

	os.environ['PYTHONWARNINGS'] = 'ignore'

	# Set the GPU
	if args.gpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	# Running train or deploy
	if args.sample:
		# TODO
		# sample
		pass
	else:
		# Set up configuration
		# Set the logger directory
		logger.auto_set_dir()

		# SyncMultiGPUTrainer(config).train()
		nr_tower = max(get_nr_gpu(), 1)
		if nr_tower == 1:
			trainer = SeparateGANTrainer(data_set, model, g_period=1, d_period=1)
		else:
			trainer = MultiGPUGANTrainer(nr_tower, data_set, model)
		trainer.train_with_defaults(
			callbacks=[
				PeriodicTrigger(ModelSaver(), every_k_epochs=50),
				ClipCallback(),
				ScheduledHyperParamSetter('learning_rate', 
					[(0, 2e-4), (100, 1e-4), (200, 2e-5), (300, 1e-5), (400, 2e-6), (500, 1e-6)], interp='linear'),
				PeriodicTrigger(VisualizeRunner(), every_k_epochs=5),
				],
			session_init=SaverRestore(args.load) if args.load else None, 
			steps_per_epoch=data_set.size(),
			max_epoch=1000, 
		)