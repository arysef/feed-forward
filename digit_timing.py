import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # y labels are oh-encoded



print (type(mnist))

#splits data into categories
n_train = mnist.train.num_examples # 55,000
n_validation = mnist.validation.num_examples # 5000
n_test = mnist.test.num_examples # 10,000

#one input, one output, and three hidden layers (so far just numbers)
n_input = 784   # input layer (28x28 pixels)
n_hidden3 = 128 # 3rd hidden layer
n_output = 10   # output layer (0-9 digits)

#more net parameters
learning_rate = 1e-4
n_iterations = 4000
batch_size = 128
dropout = 0.5

X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])
keep_prob = tf.placeholder(tf.float32) 
"""
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden3], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
"""
weights = tf.Variable(tf.truncated_normal([n_input, n_output], stddev=0.1))

biases = tf.Variable(tf.constant(0.1, shape=[n_output]))


output_layer = tf.matmul(X, weights) + biases

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
for num_gpu in range(0,1):
  #num_gpu = 0
  #sess = tf.Session()
  config = tf.ConfigProto(
          device_count = {'GPU': num_gpu}
      )
  sess = tf.Session(config=config)
  sess.run(init)


  start = time.clock()
  # train on mini batches
  for i in range(n_iterations):
      batch_x, batch_y = mnist.train.next_batch(batch_size)
      sess.run(train_step, feed_dict={X: batch_x, Y: batch_y, keep_prob:dropout})
      # print loss and accuracy (per minibatch)
      if i%100==0:
        #print ((batch_x.shape))
        minibatch_loss, minibatch_accuracy = sess.run([cross_entropy, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob:1.0})
        #print("Iteration", str(i), "\t| Loss =", str(minibatch_loss), "\t| Accuracy =", str(minibatch_accuracy))

  end = time.clock()
  print ("Time it took to train with {} GPUs: {} seconds".format(num_gpu, end-start))
  test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob:1.0})
  print("\nAccuracy on test set:", test_accuracy)


num_images = 100
img_vec = mnist.test.images[100:100+num_images]

start = time.time()
prediction = sess.run(tf.argmax(output_layer,1), feed_dict={X: img_vec})
end = time.time()


print ("Time to predict {} images: {}".format(num_images, end-start))
