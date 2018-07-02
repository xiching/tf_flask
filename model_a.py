#defines a dataflow
import tensorflow as tf
import numpy as np

def model_a():
	"""
	model placeholder : an actual model shall be defined like some CNN/LSTM etc.
	But for the purpose of demonstration,
	all we need is a simple Graph with Variable in it
	"""
	#some supposed input
	A = tf.placeholder(tf.float32, shape=[10, 10])
	#"weights" of the model
	B = tf.Variable(tf.random_uniform(A.shape.as_list()), name="my_model_a_weight")
	#model (haha)
	C = tf.matmul(A, B)

	#define a dummy class for convenience
	class Model():
		pass

	#fake model instance for easy referencing
	model = Model()

	#give input to the Graph at this node
	model.input_placeholder = A

	#get output from the Graph at this node
	model.output_node = C
	return model
A = model_a()

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#train here to change weights or however
#training_model()
#once the model is trained, or not, save the session
saver.save(sess, "model1.ckpt")
