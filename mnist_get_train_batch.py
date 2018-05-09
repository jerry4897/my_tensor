import tensorflow as tf
import numpy
from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.001
training_epochs = 15
batch_size = 100
SUMMARY_DIR = './reladam'

MNIST = input_data.read_data_sets("./MNIST_data", one_hot=True)
#tf.reset_default_graph()

with tf.name_scope('input')as scope:
    X = tf.placeholder(tf.float32, [None, 784], name='image')
    y = tf.placeholder(tf.float32, [None, 10], name='label')

with tf.variable_scope('layer1')as scope:
    W1 = tf.get_variable("W", shape=[784,256], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([256]))
    L1 = tf.nn.relu(tf.add(tf.matmul(X, W1),b1))
    
    tf.summary.histogram("X",X)
    tf.summary.histogram("weights", W1)
    tf.summary.histogram("bias",b1)
    tf.summary.histogram("layer",L1)

with tf.variable_scope('layer2')as scope:
    W2 = tf.get_variable("W", shape=[256,256], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([256]))
    L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2),b2))
    
    tf.summary.histogram("weights", W2)
    tf.summary.histogram("bias",b2)
    tf.summary.histogram("layer",L2)

with tf.variable_scope('layer3')as scope:
    W3 = tf.get_variable("W", shape=[256,10], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([10]))
    y_ = tf.add(tf.matmul(L2, W3),b3)
    
    tf.summary.histogram("weights", W3)
    tf.summary.histogram("bias",b3)
    tf.summary.histogram("logits",y_)
    
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
tf.summary.scalar("loss", loss)

summary = tf.summary.merge_all()

def get_train_batch(batch_size):
    if not hasattr(get_train_batch, "index"):
        get_train_batch._images = MNIST.train.images
        get_train_batch._labels = MNIST.train.labels
        get_train_batch.size = MNIST.train.images.shape[0]
        get_train_batch.index = 0
        
        random_st = numpy.random.get_state()
        numpy.random.shuffle(get_train_batch._images)
        numpy.random.set_state(random_st)
        numpy.random.shuffle(get_train_batch._labels)
        
    if get_train_batch.index + 128 >= get_train_batch.size:
        train_batch = (get_train_batch._images[get_train_batch.index:get_train_batch.size,],
                       get_train_batch._labels[get_train_batch.index:get_train_batch.size,])
        get_train_batch.index = 0
        
        random_st = numpy.random.get_state()
        numpy.random.shuffle(get_train_batch._images)
        numpy.random.set_state(random_st)
        numpy.random.shuffle(get_train_batch._labels)
    else:
        start = get_train_batch.index
        get_train_batch.index += 128
        end = get_train_batch.index
        train_batch = (get_train_batch._images[start:end], get_train_batch._labels[start:end])
    return train_batch

global_step = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
    for epoch in range(training_epochs):
        total_batch = int(MNIST.train.num_examples / batch_size)
        avg_loss = 0
        
        for i in range(total_batch):
            batch_xs, batch_ys = get_train_batch(batch_size)
            feed_dict = {X:batch_xs, y:batch_ys}
            s, l, _ = sess.run([summary, loss, optimizer], feed_dict=feed_dict)
            writer.add_summary(s, global_step=global_step)
            global_step += 1
            avg_loss +=l
        print('Epoch:', '%02d'%(epoch + 1), 'loss=','{:.6f}'.format(avg_loss/total_batch))
    
    correct_prediction = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc = sess.run(accuracy, feed_dict={X: MNIST.test.images, y:MNIST.test.labels})
    print('Test accuracy:',acc)
