import tensorflow as tf
import numpy as np

from model_a import model_a
from model_b import model_b

class ModelA():
	"""
	This is a wrapper class on all the compexity that one comes across.
	Principle :
	<START>
		Define graph
		Add nodes
		Assign session to the graph
		Restore sess from saved ckpt
	<DONE>
	"""
	def __init__(self):
		#defines graph
		self.graph = tf.Graph()
		#following step is very important
		with self.graph.as_default():
			"""
			consider whatever the 'graph' has as the default graph
			within the lifetime of the instance of this class
			"""
			#add nodes
			self.model = model_a()
			#assign sess
			self.sess = tf.Session()
			#saver
			self.saver = tf.train.Saver()
			#restore the model
			self.saver.restore(self.sess, "model1.ckpt")

	def predict(self, input_vec):
		result = self.sess.run(
			self.model.output_node,
			feed_dict={
				self.model.input_placeholder:input_vec
			}
		)
		return result


class ModelB():
	"""
	This is a wrapper class on all the compexity that one comes across.
	Principle :
	<START>
		Define graph
		Add nodes
		Assign session to the graph
		Restore sess from saved ckpt
	<DONE>
	"""
	def __init__(self):
		#defines graph
		self.graph = tf.Graph()
		#following step is very important
		with self.graph.as_default():
			"""
			consider whatever the 'graph' has as the default graph
			within the lifetime of the instance of this class
			"""
			#add nodes
			self.model = model_b()
			#assign sess
			self.sess = tf.Session()
			#saver
			self.saver = tf.train.Saver()
			#restore the model
			self.saver.restore(self.sess, "model2.ckpt")

	def predict(self, input_vec):
		result = self.sess.run(
			self.model.output_node,
			feed_dict={
				self.model.input_placeholder:input_vec
			}
		)
		return result
