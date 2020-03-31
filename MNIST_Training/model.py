import tensorflow as tf
import os
import numpy as np
from config import cfg

class BigModel:
    def __init__(self, args, model_type):
        self.learning_rate = 0.001
        self.num_steps = args.num_steps
        self.batch_size = args.batch_size
        self.display_step = args.display_step
        self.num_input = 784  # MNIST data input (img shape: 28*28)
        self.num_classes = 10
        self.dropoutprob = args.dropoutprob
        self.checkpoint_dir = args.checkpoint_dir
        self.checkpoint_file = "bigmodel"
        self.temperature = args.temperature
        #self.train_flag = True
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_file + ".ckpt")
        self.log_dir = os.path.join(args.log_dir, self.checkpoint_file)
        self.model_type = model_type

        # Store layers weight & bias
        self.weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.get_variable(shape = [5, 5, 1, 32], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc1")),

            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.get_variable(shape = [5, 5, 32, 64], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc2")),

            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.get_variable(shape = [7 * 7 * 64, 1024], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wd1")),

            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd2': tf.get_variable(shape = [1024, 1024], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wd2")),

            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.get_variable(shape = [1024, self.num_classes], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "out"))

        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([32]), name="%s_%s" % (self.model_type, "bc1")),
            'bc2': tf.Variable(tf.random_normal([64]), name="%s_%s" % (self.model_type, "bc2")),
            'bd1': tf.Variable(tf.random_normal([1024]), name="%s_%s" % (self.model_type, "bd1")),
            'bd2': tf.Variable(tf.random_normal([1024]), name="%s_%s" % (self.model_type, "bd2")),
            'out': tf.Variable(tf.random_normal([self.num_classes]), name="%s_%s" % (self.model_type, "out"))
        }

        self.build_model()
        self.saver = tf.train.Saver()



    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape, dtype=tf.float32 )
        return tf.Variable(initial)
    
    def conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    
    def max_pool(self, input, k_size=1, stride=1, name=None):
        return tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1], padding='SAME',name=name)
    
    def batch_norm(self, input):
        return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3, is_training=self.train_flag, updates_collections=None)



    # Create model
    def build_model(self):
        self.X = tf.placeholder(tf.float32, [None, self.num_input], name="%s_%s" % (self.model_type, "xinput"))
        self.Y = tf.placeholder(tf.float32, [None, self.num_classes], name="%s_%s" % (self.model_type, "yinput"))
        self.keep_prob = tf.placeholder(tf.float32,
                                        name="%s_%s" % (self.model_type, "dropoutprob"))  # dropout (keep probability)
        self.softmax_temperature = tf.placeholder(tf.float32, name="%s_%s" % (self.model_type, "softmaxtemp"))
        self.train_flag = tf.placeholder(tf.bool)

        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        with tf.name_scope("%sinputreshape" % (self.model_type)), tf.variable_scope(
                        "%sinputreshape" % (self.model_type)):
            x = tf.reshape(self.X, shape=[-1, 28, 28, 1])

        # Convolution Layer
        with tf.name_scope("%sconvmaxpool" % (self.model_type)), tf.variable_scope("%sconvmaxpool" % (self.model_type)):
            conv1 = tf.nn.relu(self.batch_norm(self.conv2d(x, self.weights['wc1']) + self.biases['bc1']))
            conv1 = self.max_pool(conv1, 2, 2)

            # Convolution Layer
            conv2 = tf.nn.relu(self.batch_norm(self.conv2d(conv1, self.weights['wc2']) + self.biases['bc2']))
            conv2 = self.max_pool(conv2, 2, 2)


        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        conv2 = tf.reshape(conv2, [-1, 7*7*64])
        with tf.name_scope("%sfclayer" % (self.model_type)), tf.variable_scope("%sfclayer" % (self.model_type)):
            fc1 = tf.nn.relu(self.batch_norm(tf.matmul(conv2,self.weights['wd1']) + self.biases['bd1']))
            # Apply Dropout
            fc1 = tf.nn.dropout(fc1, self.keep_prob)

            fc2 = tf.nn.relu(self.batch_norm(tf.matmul(fc1,self.weights['wd2']) + self.biases['bd2']))
            # Apply Dropout
            fc2 = tf.nn.dropout(fc2, self.keep_prob)


            # Output, class prediction
            logits = tf.add(tf.matmul(fc2, self.weights['out']), self.biases['out']) / self.softmax_temperature

        with tf.name_scope("%sprediction" % (self.model_type)), tf.variable_scope("%sprediction" % (self.model_type)):
            self.prediction = tf.nn.softmax(logits)
            # Evaluate model
            correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        with tf.name_scope("%soptimization" % (self.model_type)), tf.variable_scope(
                        "%soptimization" % (self.model_type)):
            # Define loss and optimizer
            self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=self.Y))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss_op)

        with tf.name_scope("%ssummarization" % (self.model_type)), tf.variable_scope(
                        "%ssummarization" % (self.model_type)):
            tf.summary.scalar("loss", self.loss_op)
            # Create a summary to monitor accuracy tensor
            tf.summary.scalar("accuracy", self.accuracy)

            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)

            # Merge all summaries into a single op

            # If using TF 1.6 or above, simply use the following merge_all function
            # which supports scoping
            # self.merged_summary_op = tf.summary.merge_all(scope=self.model_type)

            # Explicitly using scoping for TF versions below 1.6

            def mymergingfunction(scope_str):
                with tf.name_scope("%s_%s" % (self.model_type, "summarymerger")), tf.variable_scope(
                                "%s_%s" % (self.model_type, "summarymerger")):
                    from tensorflow.python.framework import ops as _ops
                    key = _ops.GraphKeys.SUMMARIES
                    summary_ops = _ops.get_collection(key, scope=scope_str)
                    if not summary_ops:
                        return None
                    else:
                        return tf.summary.merge(summary_ops)

            self.merged_summary_op = mymergingfunction(self.model_type)

    def start_session(self):
        self.sess = tf.Session()

    def close_session(self):
        self.sess.close()

    def train(self, dataset):

        # Initialize the variables (i.e. assign their default value)
        self.sess.run(tf.global_variables_initializer())

        print("Starting Training")

        train_data = dataset.get_train_data()
        train_summary_writer = tf.summary.FileWriter(self.log_dir, graph=self.sess.graph)
        max_accuracy = 0
        teacher_acc = []

        for step in range(1, self.num_steps + 1):
            batch_x, batch_y = train_data.next_batch(self.batch_size)
            _, summary = self.sess.run([self.train_op, self.merged_summary_op],
                                       feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: self.dropoutprob,
                                                  self.softmax_temperature: self.temperature,self.train_flag: True})
            if (step % self.display_step) == 0 or step == 1:
                # Calculate Validation loss and accuracy
                validation_x, validation_y = dataset.get_validation_data()
                loss, acc = self.sess.run([self.loss_op, self.accuracy], feed_dict={self.X: validation_x,
                                                                                    self.Y: validation_y,
                                                                                    self.keep_prob: 1.0,
                                                                                    self.train_flag: False,
                                                                                    self.softmax_temperature: 1.0})
                teacher_acc.append(acc)
                if acc > max_accuracy:
                    save_path = self.saver.save(self.sess, self.checkpoint_path)
                    print("Model Checkpointed to %s " % (save_path))

                print("Step " + str(step) + ", Validation Loss= " + "{:.4f}".format(
                    loss) + ", Validation Accuracy= " + "{:.3f}".format(acc))
        else:
            # Final Evaluation and checkpointing before training ends
            validation_x, validation_y = dataset.get_validation_data()
            loss, acc = self.sess.run([self.loss_op, self.accuracy], feed_dict={self.X: validation_x,
                                                                                self.Y: validation_y,
                                                                                self.keep_prob: 1.0,
                                                                                self.train_flag: False,
                                                                                self.softmax_temperature: 1.0})

            print("weights of final layer")


            if acc > max_accuracy:
                save_path = self.saver.save(self.sess, self.checkpoint_path)
                print("Model Checkpointed to %s " % (save_path))


        train_summary_writer.close()

        print("Optimization Finished!")

    def predict(self, data_X, temperature=1.0):
        return self.sess.run(self.prediction,
                             feed_dict={self.X: data_X, self.keep_prob: 1.0, self.train_flag: False, self.softmax_temperature: temperature})

    def run_inference(self, dataset):
        test_images, test_labels = dataset.get_test_data()
        print("Testing Accuracy:", self.sess.run(self.accuracy, feed_dict={self.X: test_images,
                                                                           self.Y: test_labels,
                                                                           self.keep_prob: 1.0,
                                                                           self.train_flag: False,
                                                                           self.softmax_temperature: 1.0
                                                                           }))

    def inference(self, dataset,levels,stddevVar, load_path ):

        with tf.Session() as self.sess:
            ckpt = tf.train.get_checkpoint_state(load_path)
            #if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

            def run_inference_w_variation(dataset,level,stddevVar):
                test_images, test_labels = dataset.get_test_data()
                rramTensors = [v for v in tf.trainable_variables()]
                #quantize levels
                if level == 0:
                    print("No quantization, Testing in Full Precision")
                else: 
                    quantize(rramTensors, levels = level)
                #Add device variations
                if stddevVar != 0.0:
                    allParameters = [v.eval() for v in rramTensors]
                    allShapes = [v.get_shape().as_list() for v in rramTensors]
                    for i in range(len(allParameters)):
                        if ('beta' not in rramTensors[i].name) or ('gamma' not in rramTensors[i].name):
                            print("rramTensors name", rramTensors[i].name)
                            param = allParameters[i]*np.exp(np.random.normal(0, stddevVar, allShapes[i]))
                            signMat = np.ones(param.shape, dtype=np.float32)
                            signMat[np.where(param < 0.0)] = -1.0
                            param = np.absolute(param)
                            param[np.where(param > cfg.RRAM.SA0_VAL)] = cfg.RRAM.SA0_VAL
                            param[np.where(param < cfg.RRAM.SA1_VAL)] = cfg.RRAM.SA1_VAL
                            param = param*signMat
                            rramTensors[i].load(param)		
                        else:
                            print('Not adding write variation for ', rramTensors[i].name)
                acc =  self.sess.run(self.accuracy, feed_dict={self.X: test_images,
                                                               self.Y: test_labels,
                                                               self.train_flag: False,
                                                               self.keep_prob: 1.0,
                                                               self.softmax_temperature: 1.0
                                                               })
          
                print("Testing accuracy", acc)
                return acc

            def quantize(rramTensors, levels=32):
                allParameters = [v.eval() for v in rramTensors]
                allShapes = [v.get_shape().as_list() for v in rramTensors]
                for i in range(len(allParameters)):
                    param = allParameters[i]
                    signMat = np.ones(param.shape, dtype=np.float32)
                    signMat[np.where(param < 0.0)] = -1.0
                    param = np.absolute(param)
                    param[np.where(param < cfg.RRAM.SA1_VAL)] = 0.0
                    param = (cfg.RRAM.SA0_VAL-cfg.RRAM.SA1_VAL)*np.ceil(param*levels)/levels + cfg.RRAM.SA1_VAL
                    param[np.where(param > cfg.RRAM.SA0_VAL)] = cfg.RRAM.SA0_VAL
                    param[np.where(param < cfg.RRAM.SA1_VAL)] = cfg.RRAM.SA1_VAL
                    param = param*signMat
                    rramTensors[i].load(param)

            print("Teacher Model accuracy with quantization level and variation %s %s" %(levels, stddevVar))
            acc = run_inference_w_variation(dataset,levels,stddevVar=stddevVar)
            return acc





    def load_model_from_file(self, load_path):
        ckpt = tf.train.get_checkpoint_state(load_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            self.sess.run(tf.global_variables_initializer())


class SmallModel:
    def __init__(self, args, model_type):
        self.learning_rate = 0.001
        self.num_steps = args.num_steps
        self.batch_size = args.batch_size
        self.display_step = args.display_step
        self.n_hidden_1 = 256  # 1st layer number of neurons
        self.n_hidden_2 = 256  # 2nd layer number of neurons
        self.num_input = 784  # MNIST data input (img shape: 28*28)
        self.num_classes = 10
        self.temperature = args.temperature
        self.checkpoint_dir = args.checkpoint_dir
        self.checkpoint_file = "smallmodel"
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_file)
        self.max_checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_file + "max")
        self.log_dir = os.path.join(args.log_dir, self.checkpoint_file)
        self.model_type = model_type


        self.weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.get_variable(shape = [5, 5, 1, 32], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc1")),

            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.get_variable(shape = [5, 5, 32, 64], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc2")),

            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.get_variable(shape = [7 * 7 * 64, 1024], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wd1")),

            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd2': tf.get_variable(shape = [1024,1024], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wd2")),

            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.get_variable(shape = [1024, self.num_classes], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "out"))

        }

        self.weights_noise = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1_noise': tf.Variable(tf.random_normal(shape = [5, 5, 1, 4], stddev=0.1)),

            # 5x5 conv, 32 inputs, 64 outputs
            'wc2_noise': tf.Variable(tf.random_normal(shape = [5, 5, 4, 8], stddev=0.1)),

            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1_noise': tf.Variable(tf.random_normal(shape = [7 * 7 * 8, 64], stddev=0.1)),

            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd2_noise': tf.Variable(tf.random_normal(shape = [64,64], stddev=0.1)),

            # 1024 inputs, 10 outputs (class prediction)
            'out_noise': tf.Variable(tf.random_normal(shape = [64, self.num_classes], stddev=0.1))

        }


        self.biases = {
            'bc1': tf.Variable(tf.random_normal([32]), name="%s_%s" % (self.model_type, "bc1")),
            'bc2': tf.Variable(tf.random_normal([64]), name="%s_%s" % (self.model_type, "bc2")),
            'bd1': tf.Variable(tf.random_normal([1024]), name="%s_%s" % (self.model_type, "bd1")),
            'bd2': tf.Variable(tf.random_normal([1024]), name="%s_%s" % (self.model_type, "bd2")),
            'out': tf.Variable(tf.random_normal([self.num_classes]), name="%s_%s" % (self.model_type, "out"))
        }


        self.build_model()

        self.saver = tf.train.Saver()

    def conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    
    def max_pool(self, input, k_size=1, stride=1, name=None):
        return tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1], padding='SAME',name=name)
    
    def batch_norm(self, input):
        return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3, is_training=self.train_flag, updates_collections=None)


    # Create model
    def build_model(self):
        self.X = tf.placeholder(tf.float32, [None, self.num_input], name="%s_%s" % (self.model_type, "xinput"))
        self.Y = tf.placeholder(tf.float32, [None, self.num_classes], name="%s_%s" % (self.model_type, "yinput"))
        self.keep_prob = tf.placeholder(tf.float32,
                                        name="%s_%s" % (self.model_type, "dropoutprob"))  # dropout (keep probability)
        self.flag = tf.placeholder(tf.bool, None, name="%s_%s" % (self.model_type, "flag"))
        self.soft_Y = tf.placeholder(tf.float32, [None, self.num_classes], name="%s_%s" % (self.model_type, "softy"))
        self.softmax_temperature = tf.placeholder(tf.float32, name="%s_%s" % (self.model_type, "softmaxtemperature"))
        self.train_flag = tf.placeholder(tf.bool)
        with tf.name_scope("%sinputreshape" % (self.model_type)), tf.variable_scope(
                        "%sinputreshape" % (self.model_type)):
            x = tf.reshape(self.X, shape=[-1, 28, 28, 1])


        with tf.name_scope("%sconvmaxpool" % (self.model_type)), tf.variable_scope("%sconvmaxpool" % (self.model_type)):
            conv1 = tf.nn.relu(self.batch_norm(self.conv2d(x, self.weights['wc1']) + self.biases['bc1']))
            conv1 = self.max_pool(conv1, 2, 2)

            # Convolution Layer
            conv2 = tf.nn.relu(self.batch_norm(self.conv2d(conv1, self.weights['wc2']) + self.biases['bc2']))
            conv2 = self.max_pool(conv2, 2, 2)


        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        conv2 = tf.reshape(conv2, [-1, 7*7*64])
        with tf.name_scope("%sfclayer" % (self.model_type)), tf.variable_scope("%sfclayer" % (self.model_type)):
            fc1 = tf.nn.relu(self.batch_norm(tf.matmul(conv2,self.weights['wd1']) + self.biases['bd1']))
            # Apply Dropout
            fc1 = tf.nn.dropout(fc1, self.keep_prob)

            fc2 = tf.nn.relu(self.batch_norm(tf.matmul(fc1,self.weights['wd2']) + self.biases['bd2']))
            # Apply Dropout
            fc2 = tf.nn.dropout(fc2, self.keep_prob)

            logits = tf.nn.relu(self.batch_norm(tf.matmul(fc2,self.weights['out']) + self.biases['out']))

        with tf.name_scope("%sprediction" % (self.model_type)), tf.variable_scope("%sprediction" % (self.model_type)):
            self.prediction = tf.nn.softmax(logits)

            self.correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        with tf.name_scope("%soptimization" % (self.model_type)), tf.variable_scope(
                        "%soptimization" % (self.model_type)):
            # Define loss and optimizer
            self.loss_op_standard = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=self.Y))

            self.total_loss = self.loss_op_standard

            self.loss_op_soft = tf.cond(self.flag,
                                        true_fn=lambda: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                            logits=logits / self.softmax_temperature, labels=self.soft_Y)),
                                        false_fn=lambda: 0.0)

            self.total_loss += tf.square(self.softmax_temperature) * self.loss_op_soft
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(self.total_loss)

        with tf.name_scope("%ssummarization" % (self.model_type)), tf.variable_scope(
                        "%ssummarization" % (self.model_type)):
            tf.summary.scalar("loss_op_standard", self.loss_op_standard)
            tf.summary.scalar("total_loss", self.total_loss)
            # Create a summary to monitor accuracy tensor
            tf.summary.scalar("accuracy", self.accuracy)

            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)

            # Merge all summaries into a single op

            # If using TF 1.6 or above, simply use the following merge_all function
            # which supports scoping
            # self.merged_summary_op = tf.summary.merge_all(scope=self.model_type)

            # Explicitly using scoping for TF versions below 1.6

            def mymergingfunction(scope_str):
                with tf.name_scope("%s_%s" % (self.model_type, "summarymerger")), tf.variable_scope(
                                "%s_%s" % (self.model_type, "summarymerger")):
                    from tensorflow.python.framework import ops as _ops
                    key = _ops.GraphKeys.SUMMARIES
                    summary_ops = _ops.get_collection(key, scope=scope_str)
                    if not summary_ops:
                        return None
                    else:
                        return tf.summary.merge(summary_ops)

            self.merged_summary_op = mymergingfunction(self.model_type)

    def start_session(self):
        self.sess = tf.Session()

    def close_session(self):
        self.sess.close()

    def train(self, dataset, teacher_model=None):
        teacher_flag = False
        if teacher_model is not None:
            teacher_flag = True

        with tf.Session() as self.sess:

            # Initialize the variables (i.e. assign their default value)
            self.sess.run(tf.global_variables_initializer())
            train_data = dataset.get_train_data()
            train_summary_writer = tf.summary.FileWriter(self.log_dir, graph=self.sess.graph)

            max_accuracy = 0

            print("Starting Training")

            def dev_step():
                validation_x, validation_y = dataset.get_validation_data()
                loss, acc = self.sess.run([self.loss_op_standard, self.accuracy], feed_dict={self.X: validation_x,
                                                                                             self.Y: validation_y,
                                                                                             # self.soft_Y: validation_y,
                                                                                             self.keep_prob: 1.0,
                                                                                             self.flag: False,
                                                                                             self.train_flag: False,
                                                                                             self.softmax_temperature: 1.0})



                print("Step " + str(step) + ", Validation Loss= " + "{:.4f}".format(
                    loss) + ", Validation Accuracy= " + "{:.3f}".format(acc))




            def dev_step_inference():
                validation_x, validation_y = dataset.get_validation_data()
                loss, acc = self.sess.run([self.loss_op_standard, self.accuracy], feed_dict={self.X: validation_x,
                                                                                             self.Y: validation_y,
                                                                                             # self.soft_Y: validation_y,
                                                                                             self.keep_prob: 1.0,
                                                                                             self.flag: False,
                                                                                             self.train_flag: False,
                                                                                             self.softmax_temperature: 1.0})

                print("Step " + str(step) + ", Validation Loss= " + "{:.4f}".format(
                    loss) + ", Validation Accuracy= " + "{:.3f}".format(acc))
                return acc



            def run_inference_w_variation(dataset,stddevVar=0.1):
                test_images, test_labels = dataset.get_test_data()
                rramTensors = [v for v in tf.trainable_variables()]
                #quantize levels
                quantize(rramTensors, levels=16)
                #Add device variations
                if stddevVar != 0.0:
                    allParameters = [v.eval() for v in rramTensors]
                    allShapes = [v.get_shape().as_list() for v in rramTensors]
                    for i in range(len(allParameters)):
                        if ('beta' not in rramTensors[i].name) or ('gamma' not in rramTensors[i].name):
                            #print("rramTensors name", rramTensors[i].name)
                            param = allParameters[i]*np.exp(np.random.normal(0, stddevVar, allShapes[i]))
                            signMat = np.ones(param.shape, dtype=np.float32)
                            signMat[np.where(param < 0.0)] = -1.0
                            param = np.absolute(param)
                            param[np.where(param > cfg.RRAM.SA0_VAL)] = cfg.RRAM.SA0_VAL
                            param[np.where(param < cfg.RRAM.SA1_VAL)] = cfg.RRAM.SA1_VAL
                            param = param*signMat
                            rramTensors[i].load(param)		
                        else:
                            print('Not adding write variation for ', rramTensors[i].name)
                #Add stuck ar fault issues
                #SF0
                addSA0(rramTensors, cfg.RRAM.SA0)
                #SF1
                addSA1(rramTensors, cfg.RRAM.SA1)
                acc = self.sess.run(self.accuracy, feed_dict={self.X: test_images,
                                                                                   self.Y: test_labels,
                                                                                   self.flag: False,
                                                                                   self.train_flag: False,
                                                                                   self.keep_prob: 1.0,
                                                                                   self.softmax_temperature: 1.0
                                                                                   })
                                                                                   
                print("Testing Accuracy:", acc)
                return acc



            def addSA1(rramTensors, percentSA1):
                """
                This function adds the SAF low defects into the crossbar.		
                """
                allParameters = [v.eval() for v in rramTensors]
                shapes = [v.get_shape().as_list() for v in rramTensors]
                minValues = []
                for i in range(len(allParameters)):
                    minValues.append(np.amin(allParameters[i]))	
                lowVal = cfg.RRAM.SA1_VAL
                for i in range(len(allParameters)):
                    if ('beta' not in rramTensors[i].name) or ('gamma' not in rramTensors[i].name):
                        param = allParameters[i]
                        dims = len(shapes)
                        if dims == 1:
                            num = int(math.ceil(shapes[i][0]*percentSA1/100.0))
                            x = np.arange(shapes[i][0])
                            random.shuffle(x)
                            for j in range(num):
                                param[x[j]] = lowVal if param[x[j]] > 0 else -1.0*lowVal
                        elif dims == 2:
                            num = int(math.ceil(shapes[i][0]*shapes[i][1]*percentSA1/100.0))
                            x = np.arange(shapes[i][0])
                            y = np.arange(shapes[i][1])
                            random.shuffle(x)
                            random.shuffle(y)
                            for j in range(num):
                                param[x[j], y[j]] = lowVal if param[x[j], y[j]] > 0 else -1.0*lowVal		
                        elif dims == 4:	
                            num = int(math.ceil(shapes[i][0]*shapes[i][1]*shapes[i][2]*shapes[i][3]*percentSA1/100.0))
                            x = np.arange(shapes[i][0])
                            y = np.arange(shapes[i][1])
                            z = np.arange(shapes[i][2])
                            k = np.arange(shapes[i][3])
                            random.shuffle(x)
                            random.shuffle(y)
                            random.shuffle(z)
                            random.shuffle(k)
                            for j in range(num):
                                param[x[j], y[j], z[j], k[j]] = lowVal if param[x[j], y[j], z[j], k[j]] > 0 else -1.0*lowVal
                        rramTensors[i].load(param)
                    else:
                        print('not adding SA1 for ', rramTensors[i].name)
            	    	
            def addSA0(rramTensors, percentSA0):
                """
                This function adds the SAF high defects into the crossbar.		
                """
                allParameters = [v.eval() for v in rramTensors]
                shapes = [v.get_shape().as_list() for v in rramTensors]
                maxValues = []
                for i in range(len(allParameters)):
                	maxValues.append(np.amax(allParameters[i]))	
                highVal = cfg.RRAM.SA0_VAL
                for i in range(len(allParameters)):
                	if ('beta' not in rramTensors[i].name) or ('gamma' not in rramTensors[i].name):
                		param = allParameters[i]
                		dims = len(shapes)
                		if dims == 1:
                			num = int(math.ceil(shapes[i][0]*percentSA0/100.0))
                			x = np.arange(shapes[i][0])
                			random.shuffle(x)
                			for j in range(num):
                				param[x[j]] = highVal
                		elif dims == 2:
                			num = int(math.ceil(shapes[i][0]*shapes[i][1]*percentSA0/100.0))
                			x = np.arange(shapes[i][0])
                			y = np.arange(shapes[i][1])
                			random.shuffle(x)
                			random.shuffle(y)
                			for j in range(num):
                				param[x[j], y[j]] = highVal			
                		elif dims == 4:	
                			num = int(math.ceil(shapes[i][0]*shapes[i][1]*shapes[i][2]*shapes[i][3]*percentSA0/100.0))
                			x = np.arange(shapes[i][0])
                			y = np.arange(shapes[i][1])
                			z = np.arange(shapes[i][2])
                			k = np.arange(shapes[i][3])
                			random.shuffle(x)
                			random.shuffle(y)
                			random.shuffle(z)
                			random.shuffle(k)
                			for j in range(num):
                				param[x[j], y[j], z[j], k[j]] = highVal	
                		rramTensors[i].load(param)	
                	else:
                		print('not adding SA0 for ', rramTensors[i].name)





            def addDeviceVariation(stddevVar=0.1):
                rramTensors = [v for v in tf.trainable_variables()]   
                if stddevVar != 0.0:
                    allParameters = [v.eval() for v in rramTensors]
                    allShapes = [v.get_shape().as_list() for v in rramTensors]
                    for i in range(len(allParameters)):
                        if ('beta' not in rramTensors[i].name) or ('gamma' not in rramTensors[i].name):
                           
                            param = allParameters[i] * np.exp(np.random.normal(0, stddevVar, allShapes[i]))
                            signMat = np.ones(param.shape, dtype=np.float32)
                            signMat[np.where(param < 0.0)] = -1.0
                            param = np.absolute(param)
                            param[np.where(param > cfg.RRAM.SA0_VAL)] = cfg.RRAM.SA0_VAL
                            param[np.where(param < cfg.RRAM.SA1_VAL)] = cfg.RRAM.SA1_VAL
                            param = param*signMat
                            rramTensors[i].load(param)		
                        else:
                            print('Not adding write variation for ', rramTensors[i].name)
                #return rramTensors


            def quantize(rramTensors, levels=32):
                allParameters = [v.eval() for v in rramTensors]
                allShapes = [v.get_shape().as_list() for v in rramTensors]
                for i in range(len(allParameters)):
                    param = allParameters[i]
                    signMat = np.ones(param.shape, dtype=np.float32)
                    signMat[np.where(param < 0.0)] = -1.0
                    param = np.absolute(param)
                    param[np.where(param < cfg.RRAM.SA1_VAL)] = 0.0
                    param = (cfg.RRAM.SA0_VAL-cfg.RRAM.SA1_VAL)*np.ceil(param*levels)/levels + cfg.RRAM.SA1_VAL
                    param[np.where(param > cfg.RRAM.SA0_VAL)] = cfg.RRAM.SA0_VAL
                    param[np.where(param < cfg.RRAM.SA1_VAL)] = cfg.RRAM.SA1_VAL
                    param = param*signMat
                    rramTensors[i].load(param)


            train_vars = tf.trainable_variables()
            quantize(train_vars, levels=10)
            addDeviceVariation(stddevVar=0.5)
            acc_list = []

            for step in range(1, self.num_steps + 1):
                batch_x, batch_y = train_data.next_batch(self.batch_size)
                soft_targets = batch_y
                if teacher_flag:
                    soft_targets = teacher_model.predict(batch_x, self.temperature)
                train_var = [v for v in tf.trainable_variables()]
                print("training variables are", train_var)
                _, summary = self.sess.run([self.train_op, self.merged_summary_op],
                                           feed_dict={self.X: batch_x,
                                                      self.Y: batch_y,
                                                      self.soft_Y: soft_targets,
                                                      self.flag: teacher_flag,
                                                      self.train_flag: True,
                                                      self.keep_prob: 0.5,
                                                      self.softmax_temperature: self.temperature}
                                           )
                if (step % self.display_step) == 0 or step == 1:
                    test_acc = dev_step_inference()

                    print("Student Model accuracy with variation")                   
                    acc = run_inference_w_variation(dataset,stddevVar=0.1)
                    if acc > max_accuracy:
                        save_path = self.saver.save(self.sess, self.checkpoint_path)
                        print("Model Checkpointed to %s " % (save_path))
                        max_accuracy = acc
                        print("max_acc", max_accuracy)
                    acc_list.append(acc)
            # Final Evaluation and checkpointing before training ends
            dev_step()
       
            train_summary_writer.close()



    def predict(self, data_X, temperature=1.0):
        return self.sess.run(self.prediction,
                             feed_dict={self.X: data_X, self.flag: False, self.train_flag: False,self.keep_prob: 1.0, self.softmax_temperature: temperature})

    def run_inference(self, dataset):
        test_images, test_labels = dataset.get_test_data()
        print("Testing Accuracy:", self.sess.run(self.accuracy, feed_dict={self.X: test_images,
                                                                           self.Y: test_labels,
                                                                           # self.soft_Y: test_labels,
                                                                           self.flag: False,
                                                                           self.train_flag: False,
                                                                           self.keep_prob: 1.0,
                                                                           self.softmax_temperature: 1.0
                                                                           }))

    def addDeviceVariation(self, rramTensors, stddevVar=0.1):
    	"""
    	This function adds write variations to trained models.
    	New params is randomly sampled from a log normal distribution centered
    	at the original value with standard deviation of stddevVar.
    	W' = W.exp(N(0, stddev)), where N is the normal distribution with 0 mean and stddev standard deviation
    	After adding the variations, the values are limitied between -1 and 1. 
    	"""
    	if stddevVar != 0.0:
    		allParameters = [v.eval() for v in rramTensors]
    		allShapes = [v.get_shape().as_list() for v in rramTensors]
    		for i in range(len(allParameters)):
    			if ('beta' not in rramTensors[i].name) or ('gamma' not in rramTensors[i].name):
    				param = allParameters[i]*np.exp(np.random.normal(0, stddevVar, allShapes[i]))
    				signMat = np.ones(param.shape, dtype=np.float32)
    				signMat[np.where(param < 0.0)] = -1.0
    				param = np.absolute(param)
    				param[np.where(param > cfg.RRAM.SA0_VAL)] = cfg.RRAM.SA0_VAL
    				param[np.where(param < cfg.RRAM.SA1_VAL)] = cfg.RRAM.SA1_VAL
    				param = param*signMat
    				rramTensors[i].load(param)		
    			else:
    				print('Not adding write variation for ', rramTensors[i].name)



    def load_model_from_file(self, load_path):
        ckpt = tf.train.get_checkpoint_state(load_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            self.sess.run(tf.global_variables_initializer())


    def inference(self, dataset,levels,stddevVar, load_path ):

        with tf.Session() as self.sess:
            ckpt = tf.train.get_checkpoint_state(load_path)
            #if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

            def run_inference_w_variation(dataset,level,stddevVar):
                test_images, test_labels = dataset.get_test_data()
                rramTensors = [v for v in tf.trainable_variables()]
                #quantize levels
                if level == 0:
                    print("No quantization, Testing in Full Precision")
                else: 
                    quantize(rramTensors, levels = level)
                #Add device variations
                if stddevVar != 0.0:
                    allParameters = [v.eval() for v in rramTensors]
                    allShapes = [v.get_shape().as_list() for v in rramTensors]
                    for i in range(len(allParameters)):
                        if ('beta' not in rramTensors[i].name) or ('gamma' not in rramTensors[i].name):
                            print("rramTensors name", rramTensors[i].name)
                            param = allParameters[i]*np.exp(np.random.normal(0, stddevVar, allShapes[i]))
                            signMat = np.ones(param.shape, dtype=np.float32)
                            signMat[np.where(param < 0.0)] = -1.0
                            param = np.absolute(param)
                            param[np.where(param > cfg.RRAM.SA0_VAL)] = cfg.RRAM.SA0_VAL
                            param[np.where(param < cfg.RRAM.SA1_VAL)] = cfg.RRAM.SA1_VAL
                            param = param*signMat
                            rramTensors[i].load(param)		
                        else:
                            print('Not adding write variation for ', rramTensors[i].name)
                acc =  self.sess.run(self.accuracy, feed_dict={self.X: test_images,
                                                               self.Y: test_labels,
                                                               self.train_flag: False,
                                                               self.keep_prob: 1.0,
                                                               self.softmax_temperature: 1.0
                                                               })
          
                print("weights of final layer")
                print("Testing accuracy", acc)
                return acc

            def quantize(rramTensors, levels=32):
                allParameters = [v.eval() for v in rramTensors]
                allShapes = [v.get_shape().as_list() for v in rramTensors]
                for i in range(len(allParameters)):
                    param = allParameters[i]
                    signMat = np.ones(param.shape, dtype=np.float32)
                    signMat[np.where(param < 0.0)] = -1.0
                    param = np.absolute(param)
                    param[np.where(param < cfg.RRAM.SA1_VAL)] = 0.0
                    param = (cfg.RRAM.SA0_VAL-cfg.RRAM.SA1_VAL)*np.ceil(param*levels)/levels + cfg.RRAM.SA1_VAL
                    param[np.where(param > cfg.RRAM.SA0_VAL)] = cfg.RRAM.SA0_VAL
                    param[np.where(param < cfg.RRAM.SA1_VAL)] = cfg.RRAM.SA1_VAL
                    param = param*signMat
                    rramTensors[i].load(param)

            print("Student Model accuracy with quantization level and variation %s %s" %(levels, stddevVar))
            acc = run_inference_w_variation(dataset,levels,stddevVar=stddevVar)
            return acc
