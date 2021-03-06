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


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data
x_data = data[:, 0:4].astype('f4')  # the samples are the four first rows of data
y_data = one_hot(data[:, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code

x_train=x_data[0:(int)(0.7*len(x_data)),:]
y_train=y_data[0:(int)(0.7*len(y_data)),:]

x_valid=x_data[(int)(0.7*len(x_data)):(int)(0.85*len(x_data)),:]
y_valid=y_data[(int)(0.7*len(x_data)):(int)(0.85*len(x_data)),:]

x_test=x_data[(int)(0.85*len(x_data)):,:]
y_test=y_data[(int)(0.85*len(x_data)):,:]

print "\nSome samples..."
for i in range(20):
    print x_data[i], " -> ", y_data[i]
print

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20
error1=0;
epoca=0;
array=[]
errorAnterior = 2000
while True:
    for jj in xrange(len(x_train) / batch_size):
        batch_xs = x_train[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_train[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    error = sess.run(loss, feed_dict={x: x_valid, y_: y_valid})

    print "Epoch #:", epoca, "Error: ", error
    result = sess.run(y, feed_dict={x: batch_xs})
    for b, r in zip(batch_ys, result):
        if b.argmax() != r.argmax():
            error1 += 1
        print b, "-->", r
    print ("Errores:", error1)
    error1 = 0;
    epoca += 1
    print "----------------------------------------------------------------------------------"

    if(abs(errorAnterior-error) < 0.0001 ):
        break
    array.append(error)
    errorAnterior=error

plt.plot(array)
plt.show()


print "----------------------"
print "   Start test...      "
print "----------------------"

resultTest=sess.run(y, feed_dict={x: x_test})
errorsTest=0
total=0
for b, r in zip(y_test, resultTest):
    if b.argmax() != r.argmax():
        errorsTest += 1
    total+=1


print('Fallos:',errorsTest)
total=(1-float(errorsTest/total))*100
print('Porcentaje aciertos:', total)
