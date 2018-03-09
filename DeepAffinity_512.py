from Utility import * 


class Model(ModelDesc):
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

		# Calculate affinity
		pa = toRangeTanh(seg_to_aff_op(toMaxLabels(pl, factor=MAX_LABEL)),  factor=1.0) # Calculate the affinity 	#0, 1

		with tf.variable_scope('gen'):
			pia, _  = self.generator(pi, last_dim=3)

		# Calculate the loss
		losses = []
		# with tf.name_scope('aff_loss'):		
		# 	# affnt_il = tf.reduce_mean(tf.abs(pa - pia_), name='affnt_il')
		# 	aff_ia = tf.reduce_mean(tf.subtract(binary_cross_entropy(toMaxLabels(pa, factor=1.0), toMaxLabels(pia, factor=1.0)), 
		# 					   					  dice_coe(toMaxLabels(pa, factor=1.0), toMaxLabels(pia, factor=1.0), axis=[0,1,2,3], loss_type='jaccard')))
		# 	# affnt_il = tf.reduce_mean(tf.subtract(binary_cross_entropy(pa, pia_), 
		# 	# 					   dice_coe(pa, pia_, axis=[0,1,2,3], loss_type='jaccard')))
		# 	losses.append(aff_ia)
		# 	add_moving_summary(aff_ia)

		with tf.name_scope('abs_loss'):		
			abs_ia = tf.reduce_mean(tf.abs(pa - pia), name='abs_loss')
			losses.append(abs_ia)
			add_moving_summary(abs_ia)	

		self.cost = tf.reduce_sum(losses)
		# Visualization
		#Segmentation
		viz = tf.concat([tf.concat([pi, pa [:,:,:,0:1], pa [:,:,:,1:2], pa [:,:,:,2:3]], 2), 
						 tf.concat([pl, pia[:,:,:,0:1], pia[:,:,:,1:2], pia[:,:,:,2:3]], 2), 
						 ], 1)

		viz = tf_2imag(viz)
		viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
		tf.summary.image('colorized', viz, max_outputs=50)


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
	parser.add_argument('--gpu',    	default='0', help='comma seperated list of GPU(s) to use.')
	parser.add_argument('--data',  default='data/Kasthuri15/3D/', required=True, 
									help='Data directory, contain trainA/trainB/validA/validB')
	parser.add_argument('--load',   help='Load the model path')
	parser.add_argument('--sample', help='Run the deployment on an instance',
									action='store_true')

	args = parser.parse_args()
	# python Exp_FusionNet2D_-VectorField.py --gpu='0' --data='arranged/'

	
	train_ds = get_data(args.data, isTrain=True, isValid=False, isTest=False)
	valid_ds = get_data(args.data, isTrain=False, isValid=True, isTest=False)
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

		session_init = SaverRestore(args.load) if args.load else None, 

		# Set up configuration
		config = TrainConfig(
			model           =   model, 
			dataflow        =   train_ds,
			callbacks       =   [
				PeriodicTrigger(ModelSaver(), every_k_epochs=50),
				PeriodicTrigger(VisualizeRunner(), every_k_epochs=5),
				# PeriodicTrigger(InferenceRunner(ds_valid, [ScalarStats('loss_membr')]), every_k_epochs=5),
				ScheduledHyperParamSetter('learning_rate', [(0, 2e-4), (100, 1e-4), (200, 1e-5), (300, 1e-6)], interp='linear')
				# ScheduledHyperParamSetter('learning_rate', [(30, 6e-6), (45, 1e-6), (60, 8e-7)]),
				# HumanHyperParamSetter('learning_rate'),
				],
			max_epoch       =   500, 
			session_init    =    SaverRestore(args.load) if args.load else None,
			#nr_tower        =   max(get_nr_gpu(), 1)
			)
	
		# Train the model
		SyncMultiGPUTrainer(config).train()