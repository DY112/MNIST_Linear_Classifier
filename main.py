import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

global_step = tf.Variable(0,trainable=False,name='global_step')

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.random_normal([784, 300]))
b1= tf.Variable(tf.random_normal([300]))
h1 = tf.nn.relu(tf.matmul(x,W1)+b1)

W2 = tf.Variable(tf.random_normal([300,10]))
b2 = tf.Variable(tf.random_normal([10]))
y = tf.matmul(h1,W2)+b2

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,logits=y))
avc = tf.Variable(0,trainable=False,name='avc')
optimizer = tf.train.AdamOptimizer(0.01).minimize(cost, global_step=global_step)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

tf.summary.scalar('cost',cost)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs',sess.graph)

for epoch in range(100):
    avc = 0.
    batch = 100
    total_batch = mnist.train.num_examples // batch
    for it in range(total_batch):
        batch_xs , batch_ys = mnist.train.next_batch(100)
        _,c,summary = sess.run([optimizer,cost,merged],feed_dict={x:batch_xs, y_:batch_ys})

        avc += c / total_batch
        writer.add_summary(summary,global_step=sess.run(global_step))
    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avc))
