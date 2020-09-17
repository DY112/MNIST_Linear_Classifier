import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
flags.DEFINE_integer('input_size', 784, 'flatten MNIST image size')
flags.DEFINE_integer('hidden_size', 300, 'hidden layer size')
flags.DEFINE_integer('train_epoch', 100, 'total training epoch')
flags.DEFINE_integer('batch_size', 100, 'training batch size')
flags.DEFINE_integer('output_size', 10, 'output label size')

global_step = tf.Variable(0,trainable=False,name='global_step')

x = tf.placeholder(tf.float32, [None, FLAGS.input_size])
y_ = tf.placeholder(tf.float32, [None, FLAGS.output_size])

W1 = tf.Variable(tf.random_normal([FLAGS.input_size, FLAGS.hidden_size]))
b1 = tf.Variable(tf.random_normal([FLAGS.hidden_size]))
h1 = tf.nn.relu(tf.matmul(x,W1)+b1)

W2 = tf.Variable(tf.random_normal([FLAGS.hidden_size,FLAGS.output_size]))
b2 = tf.Variable(tf.random_normal([FLAGS.output_size]))
y = tf.matmul(h1, W2)+b2

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,logits=y))
optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cost, global_step=global_step)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

tf.summary.scalar('cost',cost)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs',sess.graph)

saver = tf.train.Saver(tf.global_variables())

for epoch in range(FLAGS.train_epoch):
    avc = 0.
    batch = FLAGS.batch_size
    total_batch = mnist.train.num_examples // batch
    for it in range(total_batch):
        batch_xs , batch_ys = mnist.train.next_batch(FLAGS.batch_size)
        _,c,summary = sess.run([optimizer,cost,merged],feed_dict={x:batch_xs, y_:batch_ys})

        avc += c / total_batch
        writer.add_summary(summary,global_step=sess.run(global_step))

    if epoch % 100 == 99:
        saver.save(sess,'./save/model',sess.run(global_step))
    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avc))
