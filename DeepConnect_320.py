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
			]

	def _build_graph(self, inputs):
		G = tf.get_default_graph() # For round

		pi, pm, pl = inputs

		pi = tf_2tanh(pi)
		pm = tf_2tanh(pm)
		pl = tf_2tanh(pl)


		with tf.variable_scope('gen'):
			with tf.variable_scope('aff'):
				pia, _  = self.generator(pi, last_dim=3)
			with tf.variable_scope('lbl'):
				pil, _  = self.generator(pi, last_dim=1)
			
		# 
		with G.gradient_override_map({"Round": "Identity", "SegToAff": "Identity", "AffToSeg": "Identity"}):
			pa   = tf_2tanh(seg_to_aff_op(toMaxLabels(pl,  factor=MAX_LABEL),  name='pa'),   maxVal=1.0) # Calculate the affinity 	#0, 1
			pila = tf_2tanh(seg_to_aff_op(toMaxLabels(pil, factor=MAX_LABEL),  name='pila'), maxVal=1.0) # Calculate the affinity 	#0, 1

			pial = toRangeTanh(aff_to_seg_op(tf_2imag(pia, maxVal=1.0), name='pial'), factor=MAX_LABEL) # Calculate the segmentation

			pil_ = (pial + pil) / 2.0 # Return the result
			pia_ = (pila + pia) / 2.0	

		pil = tf.identity(pil, name='pil')
		pia = tf.identity(pia, name='pia')
		pial = tf.identity(pial, name='pial')
		pila = tf.identity(pila, name='pila')
		pia_ = tf.identity(pia_, name='pia_')
		pil_ = tf.identity(pil_, name='pil_')

		# Calculate the loss
		losses = []		
		with tf.name_scope('rand_loss'):
			rand_il   = tf.reduce_mean(tf_rand_score(toMaxLabels(pl,   factor=MAX_LABEL), 
													 toMaxLabels(pil , factor=MAX_LABEL)), name='rand_il')
			rand_ial  = tf.reduce_mean(tf_rand_score(toMaxLabels(pl,   factor=MAX_LABEL), 
													 toMaxLabels(pial, factor=MAX_LABEL)), name='rand_ial')
			rand_il_  = tf.reduce_mean(tf_rand_score(toMaxLabels(pl,   factor=MAX_LABEL), 
													 toMaxLabels(pil_, factor=MAX_LABEL)), name='rand_il_')

			losses.append(rand_il)
			losses.append(rand_ial)
			losses.append(rand_il_)
			add_moving_summary(rand_il)
			add_moving_summary(rand_ial)
			add_moving_summary(rand_il_)


		with tf.name_scope('aff_loss'):		
			aff_ia  = tf.identity(tf.subtract(binary_cross_entropy(tf_2imag(pa, maxVal=1.0), tf_2imag(pia, maxVal=1.0)), 
					    		 			  dice_coe(tf_2imag(pa, maxVal=1.0), tf_2imag(pia, maxVal=1.0), axis=[0,1,2,3], loss_type='jaccard')),
								 name='aff_ia')
			aff_ila = tf.identity(tf.subtract(binary_cross_entropy(tf_2imag(pa, maxVal=1.0), tf_2imag(pila, maxVal=1.0)), 
					    		 			  dice_coe(tf_2imag(pa, maxVal=1.0), tf_2imag(pila, maxVal=1.0), axis=[0,1,2,3], loss_type='jaccard')),
								 name='aff_ila')			
			aff_ia_ = tf.identity(tf.subtract(binary_cross_entropy(tf_2imag(pa, maxVal=1.0), tf_2imag(pia_, maxVal=1.0)), 
					    		 			  dice_coe(tf_2imag(pa, maxVal=1.0), tf_2imag(pia_, maxVal=1.0), axis=[0,1,2,3], loss_type='jaccard')),
								 name='aff_ia_')
			losses.append(3e-3*aff_ia)
			losses.append(3e-3*aff_ila)
			losses.append(3e-3*aff_ia_)
			add_moving_summary(aff_ia)
			add_moving_summary(aff_ila)
			add_moving_summary(aff_ia_)

		with tf.name_scope('abs_loss'):		
			abs_ia  = tf.reduce_mean(tf.abs(pa - pia), name='abs_ia')
			abs_ila = tf.reduce_mean(tf.abs(pa - pila), name='abs_ila')
			abs_ia_ = tf.reduce_mean(tf.abs(pa - pia_), name='abs_ia_')
			losses.append(abs_ia)
			losses.append(abs_ila)
			losses.append(abs_ia_)
			add_moving_summary(abs_ia)	
			add_moving_summary(abs_ila)	
			add_moving_summary(abs_ia_)	

			abs_il  = tf.reduce_mean(tf.abs(pl - pil), name='abs_il')
			abs_ial = tf.reduce_mean(tf.abs(pl - pial), name='abs_ial')
			abs_il_ = tf.reduce_mean(tf.abs(pl - pil_), name='abs_il_')
			losses.append(abs_il)
			losses.append(abs_ial)
			losses.append(abs_il_)
			add_moving_summary(abs_il)	
			add_moving_summary(abs_ial)	
			add_moving_summary(abs_il_)	
		with tf.name_scope('res_loss'):		
			res_ial = tf.reduce_mean(tf.abs(pil - pial), name='res_ial')
			res_ila = tf.reduce_mean(tf.abs(pia - pila), name='res_ila')
			losses.append(res_ial)
			losses.append(res_ila)
			add_moving_summary(res_ial)
			add_moving_summary(res_ila)	
			

		self.cost = tf.reduce_sum(losses, name='self.cost')
		add_moving_summary(self.cost)
		# Visualization
		# Segmentation
		pz = tf.zeros_like(pi)
		viz = tf.concat([tf.concat([pi, pl,   pa [:,:,:,0:1], pa [:,:,:,1:2], pa [:,:,:,2:3]], 2), 
						 tf.concat([pz, pial, pia[:,:,:,0:1], pia[:,:,:,1:2], pia[:,:,:,2:3]], 2), 
						 tf.concat([pz, pil,  pila[:,:,:,0:1], pila[:,:,:,1:2], pila[:,:,:,2:3]], 2), 
						 tf.concat([pz, pil_, pia_[:,:,:,0:1], pia_[:,:,:,1:2], pia_[:,:,:,2:3]], 2), 
						 tf.concat([pi, pl,   pil, pial, pil_], 2), 
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
			['image_p', 'membr_p', 'label_p'], ['viz'])

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


	train_ds  = PrefetchDataZMQ(train_ds, 24)
	train_ds  = PrintData(train_ds)
	# train_ds  = QueueInput(train_ds)
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

		# session_init = SaverRestore(args.load) if args.load else None, 

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
		# trainer = SyncMultiGPUTrainerReplicated(max(get_nr_gpu(), 1))
		# launch_train_with_config(config, trainer)


