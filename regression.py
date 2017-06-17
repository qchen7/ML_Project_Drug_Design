import csv
import tensorflow as tf
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read csv file using panda
ipd = pd.read_csv("myFP_CHEMBL217_trans.csv")

##   uncomment below to remove outliers based on computation of 1.5*IQR   ##
ipd = ipd.loc[ipd.measurement_value <= 1784.5]


#Randomly split the dataset into training set and test set, using a seed for re-production
shuffled = ipd.sample(frac=1,random_state=7)
trainingSet = shuffled[0:int(0.7*len(shuffled))]
testSet = shuffled[int(0.7*len(shuffled)):]

trainingSet.head()
x_train, y_train = trainingSet.ix[:,:-1], trainingSet[['measurement_value']]
x_test, y_test = testSet.ix[:,:-1], testSet[['measurement_value']]


# gloab parameters settings, change training epochs here
total_len = x_train.shape[0]
n_input = x_train.shape[1]
disp_step = 1
training_epochs = 10000
learning_rate = 0.001
batch_size = total_len

def linear():
    # create placeholders for input and output
    X = tf.placeholder(tf.float32, [None, n_input], name="X")
    Y = tf.placeholder(tf.float32, [None, 1], name="Y")

    # create weight and bias, initialized to 0
    w = tf.Variable(tf.zeros([n_input,1]), name="weights")
    b = tf.Variable(0.0, name="bias")

    # construct model to predict Y (measurement_value)
    Y_predicted = tf.matmul(X,w) + b

    # use the square error as the cost function
    cost = tf.reduce_mean(tf.square(Y - Y_predicted, name="cost"))

    # using gradient descent with learning rate of 0.001 to minimize cost
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

    with tf.Session() as sess:

        # initialize the necessary variables, in this case, w and b
        sess.run(tf.global_variables_initializer())

        # train the model
        for i in range(training_epochs): # run 10000 epochs
            train_cost = 0.
            test_cost = 0.

            # Session runs train_op to minimize cost
            _, train_cost, train_pred = sess.run([optimizer,cost,Y_predicted], feed_dict={X: x_train, Y:y_train})
            test_cost, test_pred = sess.run([cost,Y_predicted], feed_dict={X: x_test, Y:y_test})
            if (i+1)%disp_step == 0:
                print ("Epoch:", '%05d' % (i+1), "train_cost=", \
                    "{:.9f}".format(train_cost), "test_cost=", \
                    "{:.9f}".format(test_cost))
        # record the final values of w and b and corresponding predictions
        w_value, b_value = sess.run([w, b])
        train_pred = sess.run(Y_predicted, feed_dict={X: x_train, Y: y_train})
        test_pred = sess.run(Y_predicted, feed_dict={X: x_test, Y: y_test})

    # plot the real value vs the predicted value
    plt.plot(train_pred, np.asarray(y_train), 'o')
    plt.plot(np.asarray(y_train), np.asarray(y_train), 'k-',lw=2)
    plt.xlim(0, 2000)
    plt.ylim(0, 2000)
    plt.xlabel('predicted value')
    plt.ylabel('real_value')
    plt.title('Measurement_value predicted vs real (train)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    plt.plot(test_pred, np.asarray(y_test), 'o')
    plt.plot(np.asarray(y_test), np.asarray(y_test), 'k-', lw=2)
    plt.xlim(0, 2000)
    plt.ylim(0, 2000)
    plt.xlabel('predicted value')
    plt.ylabel('real_value')
    plt.title('Measurement_value predicted vs real (test)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


    return [w_value, b_value]

def ann():

    # Network parameters
    n_hidden_1 = 2048  # 1st layer number of features
    n_hidden_2 = 1024  # 2nd layer number of features
    n_hidden_3 = 2048

    # create placeholders for input and output
    X = tf.placeholder(tf.float32, [None, n_input], name="X")
    Y = tf.placeholder(tf.float32, [None, 1], name="Y")

    # create weight and bias dictionary
    w = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], 0, 0.1)),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0, 0.1)),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], 0, 0.1)),
        'out': tf.Variable(tf.random_normal([n_hidden_3, 1], 0, 0.1))
    }
    b = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1], 0, 0.1)),
        'b2': tf.Variable(tf.random_normal([n_hidden_2], 0, 0.1)),
        'b3': tf.Variable(tf.random_normal([n_hidden_3], 0, 0.1)),
        'out': tf.Variable(tf.random_normal([1], 0, 0.1))
    }

    # construct model to predict output
    def multilayer_perceptron(x, w, b):
        # Hidden layer 1
        layer_1 = tf.add(tf.matmul(x, w['h1']), b['b1'])
        layer_1 = tf.nn.sigmoid(layer_1)

        # Hidden layer 2
        layer_2 = tf.add(tf.matmul(layer_1, w['h2']), b['b2'])
        layer_2 = tf.nn.sigmoid(layer_2)

        # Hidden layer 3
        layer_3 = tf.add(tf.matmul(layer_2, w['h3']), b['b3'])
        layer_3 = tf.nn.sigmoid(layer_3)

        # Output layer
        out_layer = tf.matmul(layer_3, w['out']) + b['out']
        return out_layer

    Y_predicted = multilayer_perceptron(X, w, b)

    # use the square error as the cost function
    cost = tf.reduce_mean(tf.square(Y - Y_predicted, name="cost"))

    # using gradient descent with learning rate to minimize cost
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    with tf.Session() as sess:
    # initialize the necessary variables, in this case, w and b
        sess.run(tf.global_variables_initializer())

    # train the model
        # training cycle to minimize cost
        for epoch in range(training_epochs):
            train_cost = 0.
            test_cost = 0.
            total_batch = int(total_len / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x = x_train[i * batch_size:(i + 1) * batch_size]
                batch_y = y_train[i * batch_size:(i + 1) * batch_size]

                # Run optimization
                sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y})

                # Compute cost
            train_cost = sess.run(cost, feed_dict={X: x_train, Y: y_train})
            test_cost = sess.run(cost, feed_dict={X: x_test, Y: y_test})

            if (epoch + 1) % disp_step == 0:
                print ("Epoch:", '%05d' % (epoch + 1), "train_cost=", \
                       "{:.9f}".format(train_cost), "test_cost=", \
                       "{:.9f}".format(test_cost))

        # record the final values of w and b and corresponding predictions
        w_value, b_value = sess.run([w, b])
        train_pred = sess.run(Y_predicted, feed_dict={X: x_train, Y: y_train})
        test_pred = sess.run(Y_predicted, feed_dict={X: x_test, Y: y_test})

    # plot the real value vs the predicted value
    plt.plot(train_pred, np.asarray(y_train), 'o')
    plt.plot(np.asarray(y_train), np.asarray(y_train), 'k-', lw=2)
    plt.xlim(0, 2000)
    plt.ylim(0, 2000)
    plt.xlabel('predicted value')
    plt.ylabel('real_value')
    plt.title('Measurement_value predicted vs real (train)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    plt.plot(test_pred, np.asarray(y_test), 'o')
    plt.plot(np.asarray(y_test), np.asarray(y_test), 'k-', lw=2)
    plt.xlim(0, 2000)
    plt.ylim(0, 2000)
    plt.xlabel('predicted value')
    plt.ylabel('real_value')
    plt.title('Measurement_value predicted vs real (test)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    return [w_value, b_value]


def cnn():
    # Parameters
    droppout_rate = 0.2

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv1d(x, W):
        return tf.nn.conv1d(x, W, stride=5, padding='SAME')

    X = tf.placeholder(tf.float32, [None, 2050], name="features")
    Y = tf.placeholder(tf.float32, [None, 1], name="measurement_value")

    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(X, [-1, 2050, 1])
    # conv1 layer
    # [filter_width, in_channels, out_channels]
    W_conv1 = weight_variable([16, 1, 5])
    b_conv1 = bias_variable([5])
    h_conv1 = tf.nn.relu(conv1d(x_image, W_conv1) + b_conv1)  # 410x5

    # conv2 layer
    # [filter_width, in_channels, out_channels]
    W_conv2 = weight_variable([16, 5, 8])
    b_conv2 = bias_variable([8])
    h_conv2 = tf.nn.relu(conv1d(h_conv1, W_conv2) + b_conv2)  # 82x8

    # fc1 layer
    W_fc1 = weight_variable([82 * 8, 128])
    b_fc1 = bias_variable([128])
    h_conv2_flat = tf.reshape(h_conv2, [-1, 82 * 8])
    h_cf1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_cf1, keep_prob)

    # fc2 layer
    W_fc2 = weight_variable([128, 1])
    b_fc2 = bias_variable([1])

    Y_predicted = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # the error between prediction and real data
    cost = tf.reduce_mean(tf.square(Y - Y_predicted, name="cost"))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for epoch in range(training_epochs):
        train_cost = 0.
        test_cost = 0.
        total_batch = int(total_len / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x = x_train[i * batch_size:(i + 1) * batch_size]
            batch_y = y_train[i * batch_size:(i + 1) * batch_size]

            # Run optimization
            sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1-droppout_rate})


        # Compute cost
        train_cost = sess.run(cost, feed_dict={X: x_train, Y: y_train, keep_prob: 1})
        test_cost = sess.run(cost, feed_dict={X: x_test, Y: y_test, keep_prob: 1})

        if (epoch + 1) % disp_step == 0:
            print ("Epoch:", '%05d' % (epoch + 1), "train_cost=", \
                   "{:.9f}".format(train_cost), "test_cost=", \
                   "{:.9f}".format(test_cost))

    # record the final predictions
    train_pred = sess.run(Y_predicted, feed_dict={X: x_train, Y: y_train, keep_prob: 1})
    test_pred = sess.run(Y_predicted, feed_dict={X: x_test, Y: y_test, keep_prob: 1})

    # plot the real value vs the predicted value
    plt.plot(train_pred, np.asarray(y_train), 'o')
    plt.plot(np.asarray(y_train), np.asarray(y_train), 'k-', lw=2)
    plt.xlim(0, 2000)
    plt.ylim(0, 2000)
    plt.xlabel('predicted value')
    plt.ylabel('real_value')
    plt.title('Measurement_value predicted vs real (train)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    plt.plot(test_pred, np.asarray(y_test), 'o')
    plt.plot(np.asarray(y_test), np.asarray(y_test), 'k-', lw=2)
    plt.xlim(0, 2000)
    plt.ylim(0, 2000)
    plt.xlabel('predicted value')
    plt.ylabel('real_value')
    plt.title('Measurement_value predicted vs real (test)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    return test_pred

# user interface: choose the model for testing
def model_choice():
    print "choose the model for testing:\n" \
          "1. linear\n" \
          "2. ann\n" \
          "3. cnn\n"
    x = input()
    if x == 1:
        linear()
    elif x == 2:
        ann()
    elif x == 3:
        cnn()

model_choice()