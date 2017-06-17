import pandas as pd
import preprocess_Jason as preprocess
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


data = pd.read_csv("myFP_217_D2.csv", header=None)

D2 = preprocess.get_data(data)

x = preprocess.get_X(D2)

value = preprocess.get_target(D2)
value = MultiLabelBinarizer().fit_transform(value)
y = pd.DataFrame(value)

x, y = shuffle(x, y, random_state=0)

x_train, x_dev, x_test = x[:int((0.7*len(x)))], x[int((0.7*len(x))):int((0.9*len(x)))], x[int((0.9*len(x))):]
y_train, y_dev, y_test = y[:int((0.7*len(x)))], y[int((0.7*len(x))):int((0.9*len(x)))], y[int((0.9*len(x))):]

# gloab parameters settings, change training epochs here
total_len = x_train.shape[0]
n_input = x_train.shape[1]
disp_step = 10
training_epochs = 5
learning_rate = 0.001
batch_size = total_len
beta = 0.001


def softmax():

    # create placeholders for input and output
    X = tf.placeholder(tf.float32, [None, n_input], name="X")
    Y = tf.placeholder(tf.float32, [None, 7], name="Y")

    # create weight and bias, initialized to 0
    w = tf.Variable(tf.zeros([n_input,7]), name="weights")
    b = tf.Variable(tf.zeros([7], name="bias"))

    logits = tf.matmul(X, w) + b

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))

    ##=====Uncomment below to perform L2 Regularization=====##
    ##regularizers = tf.nn.l2_loss(w)
    ##cost = tf.reduce_mean(cost + beta * regularizers)
    ##=====Uncomment above to perform L2 Regularization=====##

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        # initialize the necessary variables, in this case, w and b
        sess.run(tf.global_variables_initializer())

        #record train and dev_test cost during training for future use (early stopping and plot of cost)
        cost_train_dev = [[],[]]
        best_dev_accuracy = []

        #set up early stopping
        early_stopping = []


        # train the model
        # training cycle to minimize cost
        for epoch in range(training_epochs):
            total_batch = int(total_len / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x = x_train[i * batch_size:(i + 1) * batch_size]
                batch_y = y_train[i * batch_size:(i + 1) * batch_size]

                # Run optimization
                sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y})

                #Compute train cost and development test cost
            train_cost = sess.run(cost, feed_dict={X: x_train, Y: y_train})
            test_dev_cost = sess.run(cost, feed_dict={X: x_dev, Y: y_dev})
            cost_train_dev[0].append(train_cost)
            cost_train_dev[1].append(test_dev_cost)

                # Compute cost and accuracy
            train_accuracy = sess.run(accuracy, feed_dict={X: x_train, Y: y_train})
            dev_test_accuracy = sess.run(accuracy, feed_dict={X: x_dev, Y: y_dev})
            test_accuracy = sess.run(accuracy, feed_dict={X: x_test, Y: y_test})

            early_stopping.append([train_accuracy, dev_test_accuracy, test_accuracy])
            best_dev_accuracy.append(dev_test_accuracy)

            if (epoch + 1) % disp_step == 0:
                print ("Epoch:", '%05d' % (epoch + 1), "train_accuracy=", \
                       "{:.9f}".format(train_accuracy), "dev_test_accuracy=", \
                       "{:.9f}".format(dev_test_accuracy), "test_accuracy=", \
                       "{:.9f}".format(test_accuracy))

        # output results assuming overfitting and U shape on the development test error
        best_index = cost_train_dev[1].index(min(cost_train_dev[1]))
        best_index2 = best_dev_accuracy.index(max(best_dev_accuracy))
        if best_index < training_epochs - 1:
            print ("Early stopping result1: Epoch:", '%05d' % (best_index + 1), "train_accuracy=", \
                   "{:.9f}".format(early_stopping[best_index][0]), "dev_test_accuracy=", \
                   "{:.9f}".format(early_stopping[best_index][1]), "test_accuracy=", \
                   "{:.9f}".format(early_stopping[best_index][2]))
        if best_index2 < training_epochs - 1:
            print ("Early stopping result2: Epoch:", '%05d' % (best_index2 + 1), "train_accuracy=", \
                   "{:.9f}".format(early_stopping[best_index2][0]), "dev_test_accuracy=", \
                   "{:.9f}".format(early_stopping[best_index2][1]), "test_accuracy=", \
                   "{:.9f}".format(early_stopping[best_index2][2]))

        return cost_train_dev


def ann():

    # Network parameters
    n_hidden_1 = 2048  # 1st layer number of features
    n_hidden_2 = 512  # 2nd layer number of features
    n_hidden_3 = 128

    # create placeholders for input and output
    X = tf.placeholder(tf.float32, [None, n_input], name="X")
    Y = tf.placeholder(tf.float32, [None, 7], name="Y")

    # create weight and bias dictionary
    w = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], 0, 0.1)),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0, 0.1)),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], 0, 0.1)),
        'out': tf.Variable(tf.random_normal([n_hidden_3, 7], 0, 0.1))
    }
    b = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1], 0, 0.1)),
        'b2': tf.Variable(tf.random_normal([n_hidden_2], 0, 0.1)),
        'b3': tf.Variable(tf.random_normal([n_hidden_3], 0, 0.1)),
        'out': tf.Variable(tf.random_normal([7], 0, 0.1))
    }

    # construct model to predict output
    def multilayer_perceptron(x, w, b):
        # Hidden layer 1
        layer_1 = tf.add(tf.matmul(x, w['h1']), b['b1'])
        layer_1 = tf.nn.relu(layer_1)

        # Hidden layer 2
        layer_2 = tf.add(tf.matmul(layer_1, w['h2']), b['b2'])
        layer_2 = tf.nn.relu(layer_2)

        # Hidden layer 3
        layer_3 = tf.add(tf.matmul(layer_2, w['h3']), b['b3'])
        layer_3 = tf.nn.relu(layer_3)

        # Output layer
        out_layer = tf.matmul(layer_3, w['out']) + b['out']
        return out_layer

    out_layer = multilayer_perceptron(X, w, b)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=out_layer))

    ##=====Uncomment below to perform L2 Regularization=====##
    ##regularizers = tf.nn.l2_loss(w['h1']) + tf.nn.l2_loss(w['h2']) + \
    ##               tf.nn.l2_loss(w['h3']) + tf.nn.l2_loss(w['out'])
    ##cost = tf.reduce_mean(cost + beta * regularizers)
    ##=====Uncomment above to perform L2 Regularization=====##

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(out_layer, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



    with tf.Session() as sess:
        # initialize the necessary variables, in this case, w and b
        sess.run(tf.global_variables_initializer())

        #record train and dev_test cost during training for future use (early stopping and plot of cost)
        cost_train_dev = [[],[]]
        best_dev_accuracy = []

        #set up early stopping
        early_stopping = []


        # train the model
        # training cycle to minimize cost
        for epoch in range(training_epochs):
            total_batch = int(total_len / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x = x_train[i * batch_size:(i + 1) * batch_size]
                batch_y = y_train[i * batch_size:(i + 1) * batch_size]

                # Run optimization
                sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y})

                #Compute train cost and development test cost
            train_cost = sess.run(cost, feed_dict={X: x_train, Y: y_train})
            test_dev_cost = sess.run(cost, feed_dict={X: x_dev, Y: y_dev})
            cost_train_dev[0].append(train_cost)
            cost_train_dev[1].append(test_dev_cost)

                # Compute cost and accuracy
            train_accuracy = sess.run(accuracy, feed_dict={X: x_train, Y: y_train})
            dev_test_accuracy = sess.run(accuracy, feed_dict={X: x_dev, Y: y_dev})
            test_accuracy = sess.run(accuracy, feed_dict={X: x_test, Y: y_test})

            early_stopping.append([train_accuracy, dev_test_accuracy, test_accuracy])
            best_dev_accuracy.append(dev_test_accuracy)

            if (epoch + 1) % disp_step == 0:
                print ("Epoch:", '%05d' % (epoch + 1), "train_accuracy=", \
                       "{:.9f}".format(train_accuracy), "dev_test_accuracy=", \
                       "{:.9f}".format(dev_test_accuracy), "test_accuracy=", \
                       "{:.9f}".format(test_accuracy))

        # output results assuming overfitting and U shape on the development test error
        best_index = cost_train_dev[1].index(min(cost_train_dev[1]))
        best_index2 = best_dev_accuracy.index(max(best_dev_accuracy))
        if best_index < training_epochs - 1:
            print ("Early stopping result1: Epoch:", '%05d' % (best_index + 1), "train_accuracy=", \
                   "{:.9f}".format(early_stopping[best_index][0]), "dev_test_accuracy=", \
                   "{:.9f}".format(early_stopping[best_index][1]), "test_accuracy=", \
                   "{:.9f}".format(early_stopping[best_index][2]))
        if best_index2 < training_epochs - 1:
            print ("Early stopping result2: Epoch:", '%05d' % (best_index2 + 1), "train_accuracy=", \
                   "{:.9f}".format(early_stopping[best_index2][0]), "dev_test_accuracy=", \
                   "{:.9f}".format(early_stopping[best_index2][1]), "test_accuracy=", \
                   "{:.9f}".format(early_stopping[best_index2][2]))

        return cost_train_dev


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
        return tf.nn.conv1d(x, W, stride=4, padding='SAME')

    X = tf.placeholder(tf.float32, [None, 2048], name="features")
    Y = tf.placeholder(tf.float32, [None, 7], name="measurement_value")

    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(X, [-1, 2048, 1])
    # conv1 layer
    # [filter_width, in_channels, out_channels]
    W_conv1 = weight_variable([16, 1, 5])
    b_conv1 = bias_variable([5])
    h_conv1 = tf.nn.relu(conv1d(x_image, W_conv1) + b_conv1)  # 512x5

    # conv2 layer
    # [filter_width, in_channels, out_channels]
    W_conv2 = weight_variable([16, 5, 8])
    b_conv2 = bias_variable([8])
    h_conv2 = tf.nn.relu(conv1d(h_conv1, W_conv2) + b_conv2)  # 128x8

    # fc1 layer
    W_fc1 = weight_variable([128 * 8, 128])
    b_fc1 = bias_variable([128])
    h_conv2_flat = tf.reshape(h_conv2, [-1, 128 * 8])
    h_cf1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_cf1, keep_prob)

    # fc2 layer
    W_fc2 = weight_variable([128, 7])
    b_fc2 = bias_variable([7])

    Y_out = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # the error between prediction and real data

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_out))

    ##=====Uncomment below to perform L2 Regularization=====##
    ##regularizers = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + \
    ##               tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2)
    ##cost = tf.reduce_mean(cost + beta * regularizers)
    ##=====Uncomment above to perform L2 Regularization=====##

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(Y_out, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        # initialize the necessary variables, in this case, w and b
        sess.run(tf.global_variables_initializer())

        #record train and dev_test cost during training for future use (early stopping and plot of cost)
        cost_train_dev = [[],[]]
        best_dev_accuracy = []

        #set up early stopping
        early_stopping = []


        # train the model
        # training cycle to minimize cost
        for epoch in range(training_epochs):
            total_batch = int(total_len / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x = x_train[i * batch_size:(i + 1) * batch_size]
                batch_y = y_train[i * batch_size:(i + 1) * batch_size]

                # Run optimization
                sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1-droppout_rate})

                #Compute train cost and development test cost
            train_cost = sess.run(cost, feed_dict={X: x_train, Y: y_train, keep_prob: 1})
            test_dev_cost = sess.run(cost, feed_dict={X: x_dev, Y: y_dev, keep_prob: 1})
            cost_train_dev[0].append(train_cost)
            cost_train_dev[1].append(test_dev_cost)

                # Compute cost and accuracy
            train_accuracy = sess.run(accuracy, feed_dict={X: x_train, Y: y_train, keep_prob: 1})
            dev_test_accuracy = sess.run(accuracy, feed_dict={X: x_dev, Y: y_dev, keep_prob: 1})
            test_accuracy = sess.run(accuracy, feed_dict={X: x_test, Y: y_test, keep_prob: 1})

            early_stopping.append([train_accuracy, dev_test_accuracy, test_accuracy])
            best_dev_accuracy.append(dev_test_accuracy)

            if (epoch + 1) % disp_step == 0:
                print ("Epoch:", '%05d' % (epoch + 1), "train_accuracy=", \
                       "{:.9f}".format(train_accuracy), "dev_test_accuracy=", \
                       "{:.9f}".format(dev_test_accuracy), "test_accuracy=", \
                       "{:.9f}".format(test_accuracy))

        # output results assuming overfitting and U shape on the development test error
        best_index = cost_train_dev[1].index(min(cost_train_dev[1]))
        best_index2 = best_dev_accuracy.index(max(best_dev_accuracy))
        if best_index < training_epochs - 1:
            print ("Early stopping result1: Epoch:", '%05d' % (best_index + 1), "train_accuracy=", \
                   "{:.9f}".format(early_stopping[best_index][0]), "dev_test_accuracy=", \
                   "{:.9f}".format(early_stopping[best_index][1]), "test_accuracy=", \
                   "{:.9f}".format(early_stopping[best_index][2]))
        if best_index2 < training_epochs - 1:
            print ("Early stopping result2: Epoch:", '%05d' % (best_index2 + 1), "train_accuracy=", \
                   "{:.9f}".format(early_stopping[best_index2][0]), "dev_test_accuracy=", \
                   "{:.9f}".format(early_stopping[best_index2][1]), "test_accuracy=", \
                   "{:.9f}".format(early_stopping[best_index2][2]))

        return cost_train_dev




# user interface: choose the model for testing
def model_choice():
    print "choose the model for testing:\n" \
          "1. softmax\n" \
          "2. ann\n" \
          "3. cnn\n"
    x = input()
    if x == 1:
        return softmax()
    elif x == 2:
        return ann()
    elif x == 3:
        return cnn()

result = model_choice()

plt.plot(range(1,training_epochs+1), result[0], "b",label = 'train_cost')
plt.plot(range(1,training_epochs+1), result[1], "r",label = 'development_test_cost')
plt.legend()
plt.xlabel('number of epochs')
plt.ylabel('cross entropy')
plt.title('train cost vs test cost')
plt.show()
