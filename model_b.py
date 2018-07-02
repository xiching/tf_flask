#defines a dataflow
import tensorflow as tf
import numpy as np

def model_b():
	"""
	model placeholder : an actual model shall be defined, like some CNN/LSTM etc.
	But for the purpose of demonstration,
	all we need is a simple Graph with Variable in it
	"""
	#some supposed input
	P = tf.placeholder(tf.float32, shape=(4,4))
	#"weights" of the model
	Q = tf.Variable(tf.constant(np.arange(16, dtype=np.float32).reshape(-1, 4)))
	#model (haha)
	R = tf.add(P, Q)

	#define a dummy class for convenience
	class Model():
		pass

	#fake model instance for easy referencing
	model = Model()

	#give input to the Graph at this node
	model.input_placeholder = P

	#get output from the Graph at this node
	model.output_node = R
	return model

#graph
A = model_b()

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#train here to change weights or however
#training_model()
#once the model is trained, or not, save the session
saver.save(sess, "model2.ckpt")
