import gzip
import cPickle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

#train_x, train_y = train_set


# ---------------- Visualizing some element of the MNIST dataset --------------


# TODO: the neural net!!
train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set



train_y = one_hot(train_y, 10)
valid_y = one_hot(valid_y, 10)
test_y = one_hot(test_y, 10)

x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 10)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(10, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

#loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
loss = tf.reduce_sum(tf.square(y_ - y))


train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 100
error1=0;
epoca=0;
array=[]
errorAnterior = 2000
while True:
    for jj in xrange((int)(len(train_x) / batch_size)):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    error = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})

    print "Epoch #:", epoca, "Error: ", error
    result = sess.run(y, feed_dict={x: valid_x})
    for b, r in zip(valid_y, result):
        if b.argmax() != r.argmax():
            error1 += 1
        #print b, "-->", r
    print ("Errores:", error1)
    error1 = 0;
    epoca += 1
    print "----------------------------------------------------------------------------------"

    if(abs(errorAnterior-error) < 0.05):
        break
    array.append(error)
    errorAnterior=error

print "----------------------"
print "   Start test...      "
print "----------------------"

resultTest = sess.run(y, feed_dict={x: test_x})
errorsTest = 0
total=0
for b, r in zip(test_y, resultTest):
    if b.argmax() != r.argmax():
        errorsTest += 1
    total+=1


print('Fallos:', errorsTest)
total=(1-float(errorsTest/total))*100
print('Porcentaje aciertos:', total)



plt.plot(array)
plt.show()
