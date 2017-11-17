from __future__ import print_function
import numpy as np
import tensorflow as tf
from hyperparams import Hyperparameters as hp
from util import new_conv_layer, new_fc_layer, flatten_layer

class Graph(object):
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, shape=[None, hp.img_h, hp.img_w, hp.color_channels], name="input_image_batch")
            #image tensor should be shaped as [None, img_width, img_height, channels] for convolution operations
            self.images = tf.reshape(self.x, shape=[-1, hp.img_w, hp.img_h, hp.color_channels])
            #actual label for image, used while training 
            self.y_true = tf.placeholder(tf.float32, shape=[None, hp.num_classes], name="y")
            self.y_true_class = tf.argmax(self.y_true, axis=0)
            self.layer_conv1 = new_conv_layer(input=self.images, num_input_channels=hp.color_channels,
                                                filter_size=hp.filter_size[0],
                                                num_filters=hp.num_filters[0],
                                                max_pooled=True)
            self.layer_conv2 = new_conv_layer(input=self.layer_conv1, num_input_channels=hp.num_filters[0],
                                                filter_size=hp.filter_size[1],
                                                num_filters=hp.num_filters[1],
                                                max_pooled=True)
            self.layer_conv3 = new_conv_layer(input=self.layer_conv2, num_input_channels=hp.num_filters[1],
                                                filter_size=hp.filter_size[2],
                                                num_filters=hp.num_filters[2],
                                                max_pooled=True)
            self.layer_flatten, num_features = flatten_layer(self.layer_conv3)
            self.layer_dense1 = new_fc_layer(self.layer_flatten, num_features, hp.dense_layer_size)
            self.layer_dense2 = new_fc_layer(self.layer_dense1, hp.dense_layer_size, hp.num_classes, False)

            self.y_pred = tf.nn.softmax(self.layer_dense2)
            self.y_pred_class = tf.argmax(self.y_pred, axis=0)
            self.correct_prediction = tf.equal(self.y_pred_class, self.y_true_class)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            #log the accuracy
            tf.summary.scalar("accuracy", self.accuracy)

            if is_training :
                self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.layer_dense2, labels=self.y_true)
                self.cost = tf.reduce_mean(self.cross_entropy)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.cost)
                #log the cost function 
                tf.summary.scalar("cost", self.cost)
            
            self.summary = tf.summary.merge_all()

def start_train(train_data_loader, epochs=hp.train_iters):
    g = Graph()
    with g.graph.as_default():
        saver = tf.train.Saver()
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter('summary',sess.graph)
            sess.run(tf.global_variables_initializer())
            for epoch in range(0,epochs):
                for i, data in enumerate(train_data_loader.load_data(),0):
                    images, labels = data.get_data() #labels shape [batch_size, 2]
                    images, labels = np.asarray(images), np.asarray(labels) #images shape [batch_size, img_h, img_w, color_channels]
                    feed_dict_train = {g.x: images, g.y_true: labels}
                    sess.run(g.optimizer, feed_dict_train)
                    if i % 10 == 0 and i != 0:
                        acc = sess.run(g.accuracy, feed_dict=feed_dict_train)
                        msg = "Epoch{} Optimization Iteration: {}, Accuracy: {}"
                        print(msg.format(epoch,i, acc))
                    if i % 100 == 0 and i != 0:    
                        #save model
                        merged = sess.run(g.summary, feed_dict_train)    
                        saver.save(sess, "model/model_latest.ckpt")



