# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = "train.p"
testing_file = "test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train_load, y_train_load = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


print(X_train_load.shape)
print(y_train_load.shape)

X_train_load[:,:,:,:] = (X_train_load[:,:,:,:] - 128) / 128



X_train, X_rest, y_train, y_rest = train_test_split(X_train_load, y_train_load, test_size=0.4)
X_validation, X_test, y_validation, y_test = train_test_split(X_rest, y_rest, test_size=0.5)
X_train, y_train = shuffle(X_train, y_train)

print("Number of training set =", len(X_train))
print("Number of validation set =", len(X_validation))
print("Number of test set =", len(X_test))

import tensorflow as tf
from tensorflow.contrib.layers import flatten



def variable_summaries(var,scope_name="summaries"):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(scope_name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def LeNet(x,keep_prob, fcp=(120,84)):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.

    variable_summaries(x,"x")
    with tf.name_scope("CONV1"):
        conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean=mu, stddev=sigma))
        conv1_b = tf.Variable(tf.zeros(6))
        conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    variable_summaries(conv1,"conv1")

    # SOLUTION: Activation.
    with tf.name_scope("RELU1"):
        conv1 = tf.nn.relu(conv1)
    variable_summaries(conv1,"conv1_relu1")


    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    with tf.name_scope("POOL1"):
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    variable_summaries(conv1,"pool1")

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    with tf.name_scope("CONV2"):
        conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
        conv2_b = tf.Variable(tf.zeros(16))
        conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    with tf.name_scope("RELU2"):
        conv2 = tf.nn.relu(conv2)

    with tf.name_scope("POOL2"):
        # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.name_scope("Flatten"):
        # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
        fc0 = flatten(conv2)

    with tf.name_scope("FC1"):
        # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
        fc1_W = tf.Variable(tf.truncated_normal(shape=(400, fcp[0]), mean=mu, stddev=sigma))
        fc1_b = tf.Variable(tf.zeros(fcp[0]))
        fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    with tf.name_scope("RELU3"):
        # SOLUTION: Activation.
        fc1 = tf.nn.relu(fc1)

    with tf.name_scope("DROP_OUT"):
        fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)

    with tf.name_scope("FC2"):
        # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
        fc2_W = tf.Variable(tf.truncated_normal(shape=(fcp[0], fcp[1]), mean=mu, stddev=sigma))
        fc2_b = tf.Variable(tf.zeros(fcp[1]))
        fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    with tf.name_scope("RELU4"):
        # SOLUTION: Activation.
        fc2 = tf.nn.relu(fc2)

    with tf.name_scope("FC5"):
        # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
        fc3_W = tf.Variable(tf.truncated_normal(shape=(fcp[1], 43), mean=mu, stddev=sigma))
        fc3_b = tf.Variable(tf.zeros(43))
        logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
keep_prob = tf.placeholder(tf.float32, ())



EPOCHS = 20
BATCH_SIZE = 128
rate = 0.001
training_keep_prob = 1
netpara = (120,84)

tf.summary.image("x", x,max_outputs=3)

logits = LeNet(x,keep_prob, netpara)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

tf.summary.scalar('accuracy', accuracy_operation)
#tf.summary.scalar("validation_accuracy", validation_accuracy)


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob:training_keep_prob})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

merged = tf.summary.merge_all()

with tf.Session() as sess:

    # sess.graph_def is the graph definition; that enables the Graph Visualizer.
    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob:1})

            if 0: #offset % (BATCH_SIZE*100) == 0.0:
                summary, Traing_loss = sess.run([merged,loss_operation], feed_dict={x: batch_x, y: batch_y, keep_prob:1})
                print("batch = {:.3f}  Traing Loss = {:.3f}".format(offset, Traing_loss))

                file_writer.add_summary(summary, i*num_examples+offset)


        Training_accuracy = evaluate(X_train, y_train)
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ... Training Accuracy= {:.3f}  Validation Accuracy = {:.3f}".format(i+1, Training_accuracy,validation_accuracy))

        #file_writer.add_summary(summary,i)

    saver.save(sess, './lenet')
    print("Model saved")
