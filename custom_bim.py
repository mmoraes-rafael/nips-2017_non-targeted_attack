from cleverhans.attacks import Attack, FastGradientMethod

from abc import ABCMeta
import numpy as np
from six.moves import xrange
import warnings
import collections

import cleverhans.utils as utils
from model import Model, CallableModelWrapper
#import cleverhans as ch

import logging

class BasicIterativeMethodMomentum(Attack):
	"""
    A variant of the Basic Iterative Method (Kurakin et al. 2016) that uses momentum in its step update. 
	The original paper used hard labels for this attack; no label smoothing.
    Paper link: https://arxiv.org/pdf/1607.02533.pdf
    """

	def __init__(self, model, back='tf', sess=None):
		"""
		Create a BasicIterativeMethod instance.
		"""
		super(BasicIterativeMethodMomentum, self).__init__(model, back, sess)
		self.feedable_kwargs = {'eps': np.float32,
                                'eps_iter': np.float32,
								'momentum': np.float32,
                                'y': np.float32,
                                'y_target': np.float32,
                                'clip_min': np.float32,
								'clip_max': np.float32}

		self.structural_kwargs = ['ord', 'nb_iter']

		if not isinstance(self.model, Model):
			self.model = CallableModelWrapper(self.model, 'probs')

	def generate(self, x, **kwargs):
		"""
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
		:momentum: (optional float) momentum weight to use in the step update
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
		"""
		import tensorflow as tf

		# Parse and save attack-specific parameters
		assert self.parse_params(**kwargs)

		# Initialize loop variables
		current_eta = 0
		last_eta = 0

		# Fix labels to the first model predictions for loss computation
		model_preds = self.model.get_probs(x)
		preds_max = tf.reduce_max(model_preds, 1, keep_dims=True)
		if self.y_target is not None:
			y = self.y_target
			targeted = True
		elif self.y is not None:
			y = self.y
			targeted = False
		else:
			y = tf.to_float(tf.equal(model_preds, preds_max))
			targeted = False

		y_kwarg = 'y_target' if targeted else 'y'
		fgm_params = {'eps': self.eps_iter, y_kwarg: y, 
						'ord': self.ord, 'clip_min': self.clip_min, 
						'clip_max': self.clip_max}

		for i in range(self.nb_iter):
			FGM = FastGradientMethod(self.model, back=self.back,
                                     sess=self.sess)
			# Compute this step's perturbation
			current_eta = FGM.generate(x + last_eta, **fgm_params) - x + self.momentum * last_eta

			# Clipping perturbation eta to self.ord norm ball
			if self.ord == np.inf:
				current_eta = tf.clip_by_value(current_eta, -self.eps, self.eps)
			elif self.ord in [1, 2]:
				reduc_ind = list(xrange(1, len(eta.get_shape())))
				if self.ord == 1:
					norm = tf.reduce_sum(tf.abs(eta),
                                         reduction_indices=reduc_ind,
                                         keep_dims=True)
				elif self.ord == 2:
					norm = tf.sqrt(tf.reduce_sum(tf.square(eta),
                                                 reduction_indices=reduc_ind,
                                                 keep_dims=True))
				eta = eta * self.eps / norm

			last_eta = current_eta

		# Define adversarial example (and clip if necessary)
		adv_x = x + current_eta
		if self.clip_min is not None and self.clip_max is not None:
			adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

		return adv_x

	def parse_params(self, eps=0.3, eps_iter=0.008, momentum=0.5, nb_iter=10, y=None,
                     ord=np.inf, clip_min=None, clip_max=None,
                     y_target=None, **kwargs):
		"""
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        Attack-specific parameters:
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
		:momentum: (optional float) momentum weight to use in the step update
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
		"""

		# Save attack-specific parameters
		self.eps = eps
		self.eps_iter = eps_iter
		self.momentum = momentum
		self.nb_iter = nb_iter
		self.y = y
		self.y_target = y_target
		self.ord = ord
		self.clip_min = clip_min
		self.clip_max = clip_max

		if self.y is not None and self.y_target is not None:
			raise ValueError("Must not set both y and y_target")
		# Check if order of the norm is acceptable given current implementation
		if self.ord not in [np.inf, 1, 2]:
			raise ValueError("Norm order must be either np.inf, 1, or 2.")
		if self.back == 'th':
			error_string = "BasicIterativeMethod is not implemented in Theano"
			raise NotImplementedError(error_string)

		return True
