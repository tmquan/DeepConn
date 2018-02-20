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
NB_FILTERS = 32  # channel size

DIMX  = 512
DIMY  = 512
DIMZ  = 5
DIMC  = 1

MAX_LABEL = 320
###############################################################################
import tensorflow as tf
import numpy as np
from malis import nodelist_like, malis_loss_weights



def seg_to_affs_op (
	seg, 
	nhood=tf.constant(malis.mknhood3d(1)), 
	name=None):
	seg = tf.squeeze(seg)
	npy_func = lambda seg, nhood: malis.seg_to_affgraph (seg, nhood).astype(np.float32)
	tf_func = tf.py_func (npy_func, [seg, nhood], [tf.float32], name=name)
	ret = tf.reshape(tf_func[0], [3, seg.shape[0], seg.shape[1], seg.shape[2]])
	ret = tf.transpose(ret, [1, 2, 3, 0])
	print ret.get_shape().as_list()
	return ret

def affs_to_seg_op (
	affs, 
	nhood=tf.constant (malis.mknhood3d(1)), 
	threshold=tf.constant(np.array([0.5])), 
	name=None):
	def npy_func (affs, nhood, threshold):
		affs = np.transpose(affs, [3, 0, 1, 2]) # zyx3 to 3zyx
		ret = malis.connected_components_affgraph ((affs > threshold[0]).astype (np.int32), nhood)[0].astype (np.int32) 
		ret = skimage.measure.label (ret).astype (np.int32)
		return ret
	print affs.get_shape().as_list()
	tf_func = tf.py_func (npy_func, [affs, nhood, threshold], [tf.int32], name=name)
	ret = tf.reshape(tf_func[0], [affs.shape[0], affs.shape[1], affs.shape[2]])
	ret = tf.expand_dims(ret, axis=-1)
	print ret.get_shape().as_list()
	return ret

# def tf_rand_score (x1, x2):
# 	def npy_func (x1, x2):
# 		ret = np.mean(1.0 - adjusted_rand_score (x1.flatten (), x2.flatten ()))
# 		return ret
# 	tf_func = tf.cast(tf.py_func (npy_func, [x1,  x2], tf.float64), tf.float32)
# 	return tf_func
def tf_rand_score (x1, x2):
	return np.mean(1.0 - adjusted_rand_score (x1.flatten (), x2.flatten ()))

def toInt32Label(label, factor=MAX_LABEL):
	result = tf.cast(label, tf.float32)
	condition  = tf.equal(result, -1.0*tf.ones_like(result))
	result = tf.where(condition, tf.zeros_like(result), result, name='removedBackground') # From -1 to 0
	result = result * factor # From 0~1 to 0~MAXLABEL
	result = tf.round(result)
	return tf.cast(result, tf.int32)


def toFloat32Label(label, factor=MAX_LABEL):
	label  = tf.cast(label, tf.float32)
	result = label / factor # From 0~MAXLABEL to 0~1
	condition = tf.equal(result, 0.0*tf.zeros_like(result))
	result = tf.where(condition, -1.0*tf.ones_like(result), result, name='addedBackground') # From -1 to 0
	return tf.cast(result, tf.float32)

###############################################################################
# class MalisWeights(object):

# 	def __init__(self, output_shape, neighborhood):
# 		self.output_shape = np.asarray(output_shape)
# 		self.neighborhood = np.asarray(neighborhood)
# 		self.edge_list = nodelist_like(self.output_shape, self.neighborhood)

# 	def get_edge_weights(self, affs, gt_affs, gt_seg):

# 		assert affs.shape[0] == len(self.neighborhood)

# 		# x_y focus pass
# 		weights_neg, neg_npairs = self.malis_pass(affs, gt_affs, gt_seg, 0.3, 1., 1., pos=0)
# 		weights_pos, pos_npairs = self.malis_pass(affs, gt_affs, gt_seg, 0.3, 1., 1., pos=1)
# 		# z focus pass
# 		# z_weights_neg, neg_npairs = self.malis_pass(affs, gt_affs, gt_seg, 1., 0.3, 0.3, pos=0)
# 		# z_weights_pos, pos_npairs = self.malis_pass(affs, gt_affs, gt_seg, 1., 0.3, 0.3, pos=1)

# 		#################################################################################
# 		# weights_neg = xy_weights_neg + z_weights_neg
# 		# weights_pos = xy_weights_pos + z_weights_pos
# 		# tot_npairs = neg_npairs + pos_npairs
# 		# norm_factor = np.prod (gt_seg.shape, dtype=np.float32)
# 		# print 'npair: ', neg_npairs, pos_npairs

# 		# ret = (weights_neg / neg_npairs) + (weights_pos / pos_npairs)
# 		# ret = weights_neg / norm_factor
# 		# weights_pos = weights_neg
# 		# ret = (weights_neg + weights_pos) / norm_factor
# 		#################################################################################
# 		# print np.sum (weights_pos > 0), np.sum (weights_neg > 0)
# 		#################################################################################
# 		# ret = weights_neg + weights_pos
# 		# minval = np.min (ret)
# 		# maxval = np.max (ret)
# 		# scaled_min = 1.0
# 		# scaled_max = 5.0
# 		# ret = (ret - minval) * (scaled_max - scaled_min) / (maxval - minval) + scaled_min
# 		# #################################################################################

# 		ret = weights_neg + weights_pos

# 		scaled_min_neg = 2.0
# 		scaled_max_neg = 10.0

# 		scaled_min_pos = 0.2
# 		scaled_max_pos = 1.0

# 		weights_pos_scaled = (weights_pos - np.min (weights_pos)) *\
# 				(scaled_max_pos - scaled_min_pos) /\
# 				(np.max (weights_pos) - np.min (weights_pos))\
# 				+ scaled_min_pos

# 		weights_neg_scaled = (weights_neg - np.min (weights_neg)) *\
# 				(scaled_max_neg - scaled_min_neg) /\
# 				(np.max (weights_neg) - np.min (weights_neg))\
# 				+ scaled_min_neg

# 		ret = weights_pos_scaled + weights_neg_scaled
		

# 		return ret , weights_pos, weights_neg

# 	def malis_pass(self, affs, gt_affs, gt_seg, z_scale, y_scale, x_scale, pos):

# 		# create a copy of the affinities and change them, such that in the
# 		#   positive pass (pos == 1): affs[gt_affs == 0] = 0
# 		#   negative pass (pos == 0): affs[gt_affs == 1] = 1

# 		pass_affs = np.copy(affs)
# 		pass_affs[gt_affs == (1 - pos)] = (1 - pos)

# 		pass_affs[0,:,:] *= z_scale
# 		pass_affs[1,:,:] *= y_scale
# 		pass_affs[2,:,:] *= x_scale

# 		weights = malis_loss_weights(
# 			gt_seg.astype(np.uint64).flatten(),
# 			self.edge_list[0].flatten(),
# 			self.edge_list[1].flatten(),
# 			pass_affs.astype(np.float32).flatten(),
# 			pos)

# 		weights = weights.reshape((-1,) + tuple(self.output_shape))
# 		assert weights.shape[0] == len(self.neighborhood)

# 		num_pairs = np.sum(weights, dtype=np.uint64)
# 		# print num_pairs
# 		# '1-pos' samples don't contribute in the 'pos' pass
# 		weights[gt_affs == (1 - pos)] = 0
# 		# print num_pairs, np.sum(weights, dtype=np.uint64), pos, np.sum (weights < 0)
# 		# print np.max (weights)
# 		# normalize
# 		weights = weights.astype(np.float32)
# 		num_pairs = np.sum(weights)
# 		# if num_pairs > 0:
# 		#     weights = weights/num_pairs

# 		return weights, num_pairs

# def malis_weights_op(affs, gt_affs, gt_seg, neighborhood, name=None):
# 	'''Returns a tensorflow op to compute just the weights of the MALIS loss.
# 	This is to be multiplied with an edge-wise base loss and summed up to create
# 	the final loss. For the Euclidean loss, use ``malis_loss_op_afft``.
# 	Args:
# 		affs (Tensor): The predicted affinities.
# 		gt_affs (Tensor): The ground-truth affinities.
# 		gt_seg (Tensor): The corresponding segmentation to the ground-truth
# 			affinities. Label 0 denotes background.
# 		neighborhood (Tensor): A list of spacial offsets, defining the
# 			neighborhood for each voxel.
# 		name (string, optional): A name to use for the operators created.
# 	Returns:
# 		A tensor with the shape of ``affs``, with MALIS weights stored for each
# 		edge.
# 	'''

# 	output_shape = gt_seg.get_shape().as_list()

# 	malis_weights = MalisWeights(output_shape, neighborhood)
# 	malis_functor = lambda affs, gt_affs, gt_seg, mw=malis_weights: \
# 		mw.get_edge_weights(affs, gt_affs, gt_seg)

# 	weights = tf.py_func(
# 		malis_functor,
# 		[affs, gt_affs, gt_seg],
# 		[tf.float32, tf.float32, tf.float32],
# 		name=name)
# 	# print weights
# 	return weights

# def malis_loss_op_afft(affs, gt_affs, gt_seg, neighborhood, name=None):
# 	'''Returns a tensorflow op to compute the MALIS loss, using the squared
# 	distance to the target values for each edge as base loss.
# 	Args:
# 		affs (Tensor): The predicted affinities.
# 		gt_affs (Tensor): The ground-truth affinities.
# 		gt_seg (Tensor): The corresponding segmentation to the ground-truth
# 			affinities. Label 0 denotes background.
# 		neighborhood (Tensor): A list of spacial offsets, defining the
# 			neighborhood for each voxel.
# 		name (string, optional): A name to use for the operators created.
# 	Returns:
# 		A tensor with one element, the MALIS loss.
# 	'''

# 	# weights, weights_pos, weights_neg = malis_weights_op(affs, gt_affs, gt_seg, neighborhood, name='malis_weights')
# 	weights, pos_weights, neg_weights = malis_weights_op(affs, gt_affs, gt_seg, neighborhood, name='malis_weights')
# 	edge_loss = tf.square(tf.subtract(gt_affs, affs))

# 	return tf.reduce_sum (tf.multiply(weights, edge_loss), name='malis_loss')
# def malis_loss_op (gt_seg, pred_seg, nhood=malis.mknhood3d(1), name=None):
# 	'''
# 		pred_seg: tensor int32
# 		gt_seg: tensor int32
# 		neighborhood: np.int32
# 	'''
# 	#convert to z, y, x
# 	gt_seg   = toInt32Label(gt_seg)
# 	pred_seg = toInt32Label(pred_seg)

# 	gt_seg   = tf.squeeze(gt_seg)
# 	pred_seg = tf.squeeze(pred_seg)

# 	pred_affs = seg_to_affs_op (pred_seg, tf.constant (nhood))
# 	gt_affs = seg_to_affs_op (gt_seg, tf.constant (nhood))
# 	return malis_loss_op_afft (pred_affs, gt_affs, gt_seg, nhood)
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

		loss_img = cvt2tanh(loss_img)
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
def cvt2tanh(x, maxVal = 255.0, name='ToRangeTanh'):
	with tf.variable_scope(name):
		return (x / maxVal - 0.5) * 2.0
###############################################################################
def cvt2imag(x, maxVal = 255.0, name='ToRangeImag'):
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
				.Conv2D('conv0', chan, padding='SAME')
				.Conv2D('conv1', chan/2, padding='SAME')
				.Conv2D('conv2', chan, padding='SAME', nl=tf.identity)
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
		pi = cvt2tanh(pi)
		pm = cvt2tanh(pm)
		pl = cvt2tanh(pl)
		ui = cvt2tanh(ui)
		um = cvt2tanh(um)
		ul = cvt2tanh(ul)


		



		with argscope([Conv2D, Deconv2D, FullyConnected],
					  W_init=tf.truncated_normal_initializer(stddev=0.02),
					  use_bias=False), \
				argscope(BatchNorm, gamma_init=tf.random_uniform_initializer()), \
				argscope([Conv2D, Deconv2D, BatchNorm], data_format='NHWC'), \
				argscope([Conv2D], dilation_rate=2):

			

			with tf.variable_scope('gen'):
				with tf.variable_scope('label'):
					pil, feat_il  = self.generator(pi, last_dim=1)
				with tf.variable_scope('affnt'):
					pia, feat_ia  = self.generator(pi, last_dim=3)
			
			# Round
			pil = toInt32Label(pil, factor=MAX_LABEL) #0 MAX
			pil = toFloat32Label(pil, factor=MAX_LABEL) # -1 1


			# 
			with tf.variable_scope('fix'):
				pa   = seg_to_affs_op(toInt32Label(pl, factor=MAX_LABEL),   name='pa') # Calculate the affinity 	#0, 1
				pila = seg_to_affs_op(toInt32Label(pil, factor=MAX_LABEL),  name='pila') # Calculate the affinity 	#0, 1
				pia  = cvt2imag(pia, maxVal=1.0) # From -1,1 to 0,1 

				pial = affs_to_seg_op(pia, name='pial') # Calculate the segmentation
				pial = toFloat32Label(pial, factor=MAX_LABEL) # -1, 1

				pil_ = (pial + pil) / 2.0 # Return the result
				pia_ = (pila + pia) / 2.0

			with tf.variable_scope('discrim'):
				# print pi 
				# print pl 
				# print pa
				# print pil_ 
				# print pia_
				print pial
				print pil
				dis_real = self.discriminator(tf.concat([pi, pl, pa], axis=-1))
				dis_fake = self.discriminator(tf.concat([pi, pil_, pia_], axis=-1))

			with tf.name_scope('GAN_loss'):
				G_loss, D_loss = self.build_losses(dis_real, dis_fake, name='gan_loss')

			with tf.name_scope('rand_loss'):
				# rand_il  = tf.reduce_mean(tf_rand_score(pl, pil_), name='rand_loss')
				rand_il  = tf.reduce_mean(tf.cast(tf.py_func (tf_rand_score, [pl, pil_], tf.float64), tf.float32), name='rand_loss')
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
					# lins = cvt2tanh(lins)
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

					lins   = cvt2tanh(lins)
					lins_z = cvt2tanh(lins_z)
					lins_y = cvt2tanh(lins_y)
					lins_x = cvt2tanh(lins_x)

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
				discrim_il  = regDLF(toInt32Label(pl, factor=MAX_LABEL), feat_il, name='discrim_il')
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

				thresholded_mag_grad_L = cvt2tanh(thresholded_mag_grad_L, maxVal=1.0)
				return thresholded_mag_grad_L

			g_il =  label_imag(pil_, name='label_il')
			g_l  =  label_imag(pl,   name='label_l')

		self.g_loss = tf.reduce_sum([
								1*(G_loss), 
								10*(recon_il), 
								10*(residual_il), 
								1*(rand_il), 
								20*(discrim_il), 
								0.002*affnt_il, 		
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
		
		viz = cvt2imag(viz)
		viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
		tf.summary.image('colorized', viz, max_outputs=50)

		# Affinity
		vis = tf.concat([pa, pila, pia, pia_ ], 2)
		
		vis = cvt2imag(vis)
		vis = tf.cast(tf.clip_by_value(vis, 0, 255), tf.uint8, name='vis')
		tf.summary.image('affinities', vis, max_outputs=50)

		# 	pil   = rounded(pil)	
		# 	with tf.variable_scope('discrim'):
		# 		l_dis_real = self.discriminator(tf.concat([pi, pl], axis=-1))
		# 		l_dis_fake = self.discriminator(tf.concat([pi, pil], axis=-1))
		
		# with tf.name_scope('Equal_L_loss'):		
		# 	equal_il = tf.reduce_mean(tf.cast(
		# 						tf.not_equal(cvt2imag(pl, maxVal=MAX_LABEL), cvt2imag(pil, maxVal=MAX_LABEL)), 
		# 						tf.float32), name='equal_il')

		# with tf.name_scope('Recon_L_loss'):		
		# 	recon_il 		= tf.reduce_mean(tf.abs(cvt2imag(pl, maxVal=MAX_LABEL) - cvt2imag(pil, maxVal=MAX_LABEL)), name='recon_il')
			
		# with tf.name_scope('GAN_loss'):
		# 	G_loss, D_loss = self.build_losses(l_dis_real, l_dis_fake, name='gan_il')

		


		# # custom loss for tf_rand_score
		# with tf.name_scope('rand_loss'):
		# 	rand_il  = tf.reduce_mean()

		# with tf.name_scope('discrim_loss'):
		# 	def regDLF(y_true, y_pred, alpha=1, beta=1, gamma=0.01, delta_v=0.5, delta_d=1.5, name='loss_discrim'):
		# 		def tf_norm(inputs, axis=1, epsilon=1e-7,  name='safe_norm'):
		# 			squared_norm 	= tf.reduce_sum(tf.square(inputs), axis=axis, keep_dims=True)
		# 			safe_norm 		= tf.sqrt(squared_norm+epsilon)
		# 			return tf.identity(safe_norm, name=name)
		# 		###


		# 		lins = tf.linspace(0.0, DIMZ*DIMY*DIMX, DIMZ*DIMY*DIMX)
		# 		lins = tf.cast(lins, tf.int32)
		# 		# lins = lins / tf.reduce_max(lins) * 255
		# 		# lins = cvt2tanh(lins)
		# 		# lins = tf.reshape(lins, tf.shape(y_true), name='lins_3d')
		# 		# print lins
		# 		lins_z = tf.div(lins,(DIMY*DIMX))
		# 		lins_y = tf.div(tf.mod(lins,(DIMY*DIMX)), DIMY)
		# 		lins_x = tf.mod(tf.mod(lins,(DIMY*DIMX)), DIMY)

		# 		lins   = tf.cast(lins  , tf.float32)
		# 		lins_z = tf.cast(lins_z, tf.float32)
		# 		lins_y = tf.cast(lins_y, tf.float32)
		# 		lins_x = tf.cast(lins_x, tf.float32)

		# 		lins   = lins 	/ tf.reduce_max(lins) * 255
		# 		lins_z = lins_z / tf.reduce_max(lins_z) * 255
		# 		lins_y = lins_y / tf.reduce_max(lins_y) * 255
		# 		lins_x = lins_x / tf.reduce_max(lins_x) * 255

		# 		lins   = cvt2tanh(lins)
		# 		lins_z = cvt2tanh(lins_z)
		# 		lins_y = cvt2tanh(lins_y)
		# 		lins_x = cvt2tanh(lins_x)

		# 		lins   = tf.reshape(lins,   tf.shape(y_true), name='lins')
		# 		lins_z = tf.reshape(lins_z, tf.shape(y_true), name='lins_z')
		# 		lins_y = tf.reshape(lins_y, tf.shape(y_true), name='lins_y')
		# 		lins_x = tf.reshape(lins_x, tf.shape(y_true), name='lins_x')

		# 		y_true = tf.reshape(y_true, [DIMZ*DIMY*DIMX])
		# 		y_pred = tf.concat([y_pred, lins, lins_z, lins_y, lins_x], axis=-1)

		# 		nDim = tf.shape(y_pred)[-1]
		# 		X = tf.reshape(y_pred, [DIMZ*DIMY*DIMX, nDim])
		# 		uniqueLabels, uniqueInd = tf.unique(y_true)

		# 		numUnique = tf.size(uniqueLabels) # Get the number of connected component

		# 		Sigma = tf.unsorted_segment_sum(X, uniqueInd, numUnique)
		# 		# ones_Sigma = tf.ones((tf.shape(X)[0], 1))
		# 		ones_Sigma = tf.ones_like(X)
		# 		ones_Sigma = tf.unsorted_segment_sum(ones_Sigma, uniqueInd, numUnique)
		# 		mu = tf.divide(Sigma, ones_Sigma)

		# 		Lreg = tf.reduce_mean(tf.norm(mu, axis=1, ord=1))

		# 		T = tf.norm(tf.subtract(tf.gather(mu, uniqueInd), X), axis = 1, ord=1)
		# 		T = tf.divide(T, Lreg)
		# 		T = tf.subtract(T, delta_v)
		# 		T = tf.clip_by_value(T, 0, T)
		# 		T = tf.square(T)

		# 		ones_Sigma = tf.ones_like(uniqueInd, dtype=tf.float32)
		# 		ones_Sigma = tf.unsorted_segment_sum(ones_Sigma, uniqueInd, numUnique)
		# 		clusterSigma = tf.unsorted_segment_sum(T, uniqueInd, numUnique)
		# 		clusterSigma = tf.divide(clusterSigma, ones_Sigma)

		# 		# Lvar = tf.reduce_mean(clusterSigma, axis=0)
		# 		Lvar = tf.reduce_mean(clusterSigma)

		# 		mu_interleaved_rep = tf.tile(mu, [numUnique, 1])
		# 		mu_band_rep = tf.tile(mu, [1, numUnique])
		# 		mu_band_rep = tf.reshape(mu_band_rep, (numUnique*numUnique, nDim))

		# 		mu_diff = tf.subtract(mu_band_rep, mu_interleaved_rep)
		# 				# Remove zero vector
		# 				# intermediate_tensor = reduce_sum(tf.abs(x), 1)
		# 				# zero_vector = tf.zeros(shape=(1,1), dtype=tf.float32)
		# 				# bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
		# 				# omit_zeros = tf.boolean_mask(x, bool_mask)
		# 		intermediate_tensor = tf.reduce_sum(tf.abs(mu_diff), 1)
		# 		zero_vector = tf.zeros(shape=(1,1), dtype=tf.float32)
		# 		bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
		# 		omit_zeros = tf.boolean_mask(mu_diff, bool_mask)
		# 		mu_diff = tf.expand_dims(omit_zeros, axis=1)
		# 		print mu_diff
		# 		mu_diff = tf.norm(mu_diff, ord=1)
		# 				# squared_norm = tf.reduce_sum(tf.square(s), axis=axis,keep_dims=True)
		# 				# safe_norm = tf.sqrt(squared_norm + epsilon)
		# 				# squared_norm = tf.reduce_sum(tf.square(omit_zeros), axis=-1,keep_dims=True)
		# 				# safe_norm = tf.sqrt(squared_norm + 1e-6)
		# 				# mu_diff = safe_norm

		# 		mu_diff = tf.divide(mu_diff, Lreg)

		# 		mu_diff = tf.subtract(2*delta_d, mu_diff)
		# 		mu_diff = tf.clip_by_value(mu_diff, 0, mu_diff)
		# 		mu_diff = tf.square(mu_diff)

		# 		numUniqueF = tf.cast(numUnique, tf.float32)
		# 		Ldist = tf.reduce_mean(mu_diff)        

		# 		# L = alpha * Lvar + beta * Ldist + gamma * Lreg
		# 		# L = tf.reduce_mean(L, keep_dims=True)
		# 		L = tf.reduce_sum([alpha*Lvar, beta*Ldist, gamma*Lreg], keep_dims=False)
		# 		print L
		# 		print Ldist
		# 		print Lvar
		# 		print Lreg
		# 		return tf.identity(L,  name=name)

		# 	discrim_il  = regDLF(cvt2imag(pl, maxVal=MAX_LABEL), feat_il, name='discrim_il')
		
		# # custom loss for membr
		# with tf.name_scope('affnt_loss'):
		# 	# def seg_to_affs_op (seg, nhood=malis.mknhood3d(1), name=None):
		# 	# 	np_func = lambda seg, nhood: malis.seg_to_affgraph (seg, nhood).astype(np.float32)
		# 	# 	tf_func = tf.py_func (np_func, [seg, nhood], [tf.float32], name=name)
		# 	# 	return tf_func[0]
		# 	# def affnt_loss(y_true, y_pred, name='affnt_loss'):
		# 	# 	loss = []
		# 	# 	afft_z_true = tf.cast(
		# 	# 					tf.equal(y_true[:-1,...], y_true[1:,...]), 
		# 	# 					tf.float32)
		# 	# 	afft_z_pred = tf.cast(
		# 	# 					tf.equal(y_pred[:-1,...], y_pred[1:,...]), 
		# 	# 					tf.float32)
		# 	# 	#loss_afft_z = tf.reduce_mean(tf.abs(afft_z_true - afft_z_pred))
		# 	# 	loss_afft_z = tf.reduce_mean(tf.subtract(binary_cross_entropy(afft_z_true, afft_z_pred), 
		# 	# 						 dice_coe(afft_z_true, afft_z_pred, axis=[0,1,2,3], loss_type='jaccard')))
		# 	# 	loss.append(loss_afft_z)

		# 	# 	afft_y_true = tf.cast(
		# 	# 					tf.equal(y_true[:,:-1,...], y_true[:,1:,...]), 
		# 	# 					tf.float32)
		# 	# 	afft_y_pred = tf.cast(
		# 	# 					tf.equal(y_pred[:,:-1,...], y_pred[:,1:,...]), 
		# 	# 					tf.float32)
		# 	# 	#loss_afft_y = tf.reduce_mean(tf.abs(afft_y_true - afft_y_pred))
		# 	# 	loss_afft_y = tf.reduce_mean(tf.subtract(binary_cross_entropy(afft_y_true, afft_y_pred), 
		# 	# 						 dice_coe(afft_y_true, afft_y_pred, axis=[0,1,2,3], loss_type='jaccard')))
		# 	# 	loss.append(loss_afft_y)

		# 	# 	afft_x_true = tf.cast(
		# 	# 					tf.equal(y_true[:,:,:-1,...], y_true[:,:,1:,...]), 
		# 	# 					tf.float32)
		# 	# 	afft_x_pred = tf.cast(
		# 	# 					tf.equal(y_pred[:,:,:-1,...], y_pred[:,:,1:,...]), 
		# 	# 					tf.float32)
		# 	# 	# loss_afft_x = tf.reduce_mean(tf.abs(afft_x_true - afft_x_pred))
		# 	# 	loss_afft_x = tf.reduce_mean(tf.subtract(binary_cross_entropy(afft_x_true, afft_x_pred), 
		# 	# 						 dice_coe(afft_x_true, afft_x_pred, axis=[0,1,2,3], loss_type='jaccard')))

		# 	# 	loss.append(loss_afft_x)

		# 	# 	return tf.reduce_mean(loss, name=name)

		# 	affnt_il  = affnt_loss(pl, pil, name='affnt_il')


		# # custom loss for label
		# with tf.name_scope('label_loss'):
		# 	def label_loss_op(y_pred_L, name='label_loss_op'):
		# 		g_mag_grad_M = tf.cast(tf.greater(y_pred_L, -1.0*tf.ones_like(y_pred_L)), tf.float32) # cvt2imag(y_grad_M, maxVal=1.0)
		# 		mag_grad_L   = magnitute_central_difference(y_pred_L, name='mag_grad_L')
		# 		cond = tf.greater(mag_grad_L, tf.zeros_like(mag_grad_L))
		# 		thresholded_mag_grad_L = tf.where(cond, 
		# 								   tf.ones_like(mag_grad_L), 
		# 								   tf.zeros_like(mag_grad_L), 
		# 								   name='thresholded_mag_grad_L')

		# 		gtv_guess = tf.multiply(g_mag_grad_M, thresholded_mag_grad_L, name='gtv_guess')
		# 		loss_gtv_guess = tf.reduce_mean(gtv_guess, name='loss_gtv_guess')
		# 		# loss_gtv_guess = tf.reshape(loss_gtv_guess, [-1])
		# 		thresholded_mag_grad_L = cvt2tanh(thresholded_mag_grad_L, maxVal=1.0)
		# 		gtv_guess = cvt2tanh(gtv_guess, maxVal=1.0)
		# 		return tf.identity(loss_gtv_guess, name=name), thresholded_mag_grad_L

		# 	label_il, g_il = label_loss_op(pil, name='label_im')
		# 	label_l,  g_l  = label_loss_op(pl, name='label_l')

		# # custom loss for malis
		# with tf.name_scope('malis_loss'):
		# 	malis_il = malis_loss_op (pl, pil, nhood=malis.mknhood3d(1), name='malis_il')



		# self.g_loss = tf.reduce_sum([
		# 						1*(G_loss), 
		# 						10*(recon_il), 
		# 						10*(equal_il), 
		# 						1e-5*(malis_il), 
		# 						1*(rand_il), 
		# 						1*(label_il), 
		# 						20*(discrim_il), 
		# 						0.002*affnt_il, 		
		# 						], name='G_loss_total')
		# self.d_loss = tf.reduce_sum([
		# 						D_loss
		# 						], name='D_loss_total')

		# wd_g = regularize_cost('gen/.*/W', 		l2_regularizer(1e-5), name='G_regularize')
		# wd_d = regularize_cost('discrim/.*/W', 	l2_regularizer(1e-5), name='D_regularize')

		# self.g_loss = tf.add(self.g_loss, wd_g, name='g_loss')
		# self.d_loss = tf.add(self.d_loss, wd_d, name='d_loss')

	

		# self.collect_variables()

		# add_moving_summary(self.d_loss, self.g_loss)
		# # with tf.name_scope('summaries'):	
		# # 	add_tensor_summary(equal_il, 		types=['scalar'], name='equal_il')
		# # 	add_tensor_summary(malis_il, 		types=['scalar'], name='malis_il')
		# # 	add_tensor_summary(recon_il, 		types=['scalar'], name='recon_il')
		# # 	add_tensor_summary(rand_il, 		types=['scalar'], name='rand_il')
		# # 	add_tensor_summary(discrim_il, 		types=['scalar'], name='discrim_il')
		# # 	add_tensor_summary(affnt_il, 		types=['scalar'], name='affnt_il')
		# # 	add_tensor_summary(label_il, 		types=['scalar'], name='label_il')
			
		# def label_imag(y_pred_L, name='label_imag'):
		# 	mag_grad_L   = magnitute_central_difference(y_pred_L, name='mag_grad_L')
		# 	cond = tf.greater(mag_grad_L, tf.zeros_like(mag_grad_L))
		# 	thresholded_mag_grad_L = tf.where(cond, 
		# 							   tf.ones_like(mag_grad_L), 
		# 							   tf.zeros_like(mag_grad_L), 
		# 							   name='thresholded_mag_grad_L')

		# 	thresholded_mag_grad_L = cvt2tanh(thresholded_mag_grad_L, maxVal=1.0)
		# 	return thresholded_mag_grad_L

		# g_il =  label_imag(pil, name='label_il')
		# g_l  =  label_imag(pl, name='label_l')
		# viz = tf.concat([tf.concat([pi, pl,  g_l], 2), 
		# 				 tf.concat([pl, pil, g_il], 2),
		# 				 ], 1)
		
		# viz = cvt2imag(viz)
		# viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
		# tf.summary.image('colorized', viz, max_outputs=50)

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