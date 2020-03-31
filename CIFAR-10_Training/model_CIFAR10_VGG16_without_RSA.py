import tensorflow as tf
import os
import numpy as np
from config import cfg
from data_utility import *
class BigModel:
    def __init__(self, args, model_type):
        #self.learning_rate = 0.001
        self.num_steps = args.num_steps
        self.batch_size = args.batch_size
        self.total_epoch = args.total_epoch
        self.display_step = args.display_step
        self.iterations = args.iterations
        self.num_input = 784  # MNIST data input (img shape: 28*28)
        self.image_size = 32
        self.num_classes = 10
        self.dropoutprob = args.dropoutprob
        self.checkpoint_dir = args.checkpoint_dir
        self.checkpoint_file = "bigmodel_CIFAR10"
        self.temperature = args.temperature
        #self.train_flag = True
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_file + ".ckpt")
        self.log_dir = os.path.join(args.log_dir, self.checkpoint_file)
        self.model_type = model_type
        self.momentum_rate = 0.9
        self.weight_decay = 0.0003

        # Store layers weight & bias
        self.weights = {
            # 5x5 conv, 1 input, 32 outputs
            #'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32]), name="%s_%s" % (self.model_type, "wc1")),
            'wc1_1': tf.get_variable(shape = [3, 3, 3, 64], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc1_1")),
            'wc1_2': tf.get_variable(shape = [3, 3, 64, 64], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc1_2")),

            # 5x5 conv, 32 inputs, 64 outputs
            #'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64]), name="%s_%s" % (self.model_type, "wc2")),
            'wc2_1': tf.get_variable(shape = [3, 3, 64, 128], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc2_1")),
            'wc2_2': tf.get_variable(shape = [3, 3, 128, 128], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc2_2")),


            'wc3_1': tf.get_variable(shape = [3, 3, 128, 256], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc3_1")),
            'wc3_2': tf.get_variable(shape = [3, 3, 256, 256], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc3_2")),
            'wc3_3': tf.get_variable(shape = [3, 3, 256, 256], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc3_3")),


            'wc4_1': tf.get_variable(shape = [3, 3, 256, 512], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc4_1")),
            'wc4_2': tf.get_variable(shape = [3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc4_2")),
            'wc4_3': tf.get_variable(shape = [3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc4_3")),

            'wc5_1': tf.get_variable(shape = [3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc5_1")),
            'wc5_2': tf.get_variable(shape = [3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc5_2")),
            'wc5_3': tf.get_variable(shape = [3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc5_3")),


            # fully connected, 7*7*64 inputs, 1024 outputs
            #'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024]), name="%s_%s" % (self.model_type, "wd1")),
            'wd1': tf.get_variable(shape = [2 * 2 * 512, 4096], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wd1")),

            # fully connected, 7*7*64 inputs, 1024 outputs
            #'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024]), name="%s_%s" % (self.model_type, "wd1")),
            'wd2': tf.get_variable(shape = [4096, 4096], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wd2")),

            # 1024 inputs, 10 outputs (class prediction)
            #'out': tf.Variable(tf.random_normal([1024, self.num_classes]), name="%s_%s" % (self.model_type, "out"))
            'out': tf.get_variable(shape = [4096, self.num_classes], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "out"))

        }

        self.biases = {
            'bc1_1': tf.Variable(tf.random_normal([64]), name="%s_%s" % (self.model_type, "bc1_1")),
            'bc1_2': tf.Variable(tf.random_normal([64]), name="%s_%s" % (self.model_type, "bc1_2")),

            'bc2_1': tf.Variable(tf.random_normal([128]), name="%s_%s" % (self.model_type, "bc2_1")),
            'bc2_2': tf.Variable(tf.random_normal([128]), name="%s_%s" % (self.model_type, "bc2_2")),

            'bc3_1': tf.Variable(tf.random_normal([256]), name="%s_%s" % (self.model_type, "bc3_1")),
            'bc3_2': tf.Variable(tf.random_normal([256]), name="%s_%s" % (self.model_type, "bc3_2")),
            'bc3_3': tf.Variable(tf.random_normal([256]), name="%s_%s" % (self.model_type, "bc3_3")),


            'bc4_1': tf.Variable(tf.random_normal([512]), name="%s_%s" % (self.model_type, "bc4_1")),
            'bc4_2': tf.Variable(tf.random_normal([512]), name="%s_%s" % (self.model_type, "bc4_2")),
            'bc4_3': tf.Variable(tf.random_normal([512]), name="%s_%s" % (self.model_type, "bc4_3")),

            'bc5_1': tf.Variable(tf.random_normal([512]), name="%s_%s" % (self.model_type, "bc5_1")),
            'bc5_2': tf.Variable(tf.random_normal([512]), name="%s_%s" % (self.model_type, "bc5_2")),
            'bc5_3': tf.Variable(tf.random_normal([512]), name="%s_%s" % (self.model_type, "bc5_3")),




            'bd1': tf.Variable(tf.random_normal([4096]), name="%s_%s" % (self.model_type, "bd1")),
            'bd2': tf.Variable(tf.random_normal([4096]), name="%s_%s" % (self.model_type, "bd2")),
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
        #self.X = tf.placeholder(tf.float32, [None, self.num_input], name="%s_%s" % (self.model_type, "xinput"))
        self.X = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name="%s_%s" % (self.model_type, "xinput"))
        self.Y = tf.placeholder(tf.float32, [None, self.num_classes], name="%s_%s" % (self.model_type, "yinput"))
        self.keep_prob = tf.placeholder(tf.float32,
                                        name="%s_%s" % (self.model_type, "dropoutprob"))  # dropout (keep probability)
        self.softmax_temperature = tf.placeholder(tf.float32, name="%s_%s" % (self.model_type, "softmaxtemp"))
        self.train_flag = tf.placeholder(tf.bool)
        self.learning_rate = tf.placeholder(tf.float32)



        # Convolution Layer
        with tf.name_scope("%sconvmaxpool" % (self.model_type)), tf.variable_scope("%sconvmaxpool" % (self.model_type)):
            #conv1 = self.conv2d(x, self.weights['wc1'], self.biases['bc1'])
            conv1_1 = tf.nn.relu(self.batch_norm(self.conv2d(self.X, self.weights['wc1_1']) + self.biases['bc1_1']))
            conv1_2 = tf.nn.relu(self.batch_norm(self.conv2d(conv1_1, self.weights['wc1_2']) + self.biases['bc1_2']))

            # Max Pooling (down-sampling)
            #conv1 = self.maxpool2d(conv1, k=2)
            conv1_pool = self.max_pool(conv1_2, 2, 2)

            # Convolution Layer
            #conv2 = self.conv2d(conv1, self.weights['wc2'], self.biases['bc2'])
            conv2_1 = tf.nn.relu(self.batch_norm(self.conv2d(conv1_pool, self.weights['wc2_1']) + self.biases['bc2_1']))
            conv2_2 = tf.nn.relu(self.batch_norm(self.conv2d(conv2_1, self.weights['wc2_2']) + self.biases['bc2_2']))

            # Max Pooling (down-sampling)
            #conv2 = self.maxpool2d(conv2, k=2)
            conv2_pool = self.max_pool(conv2_2, 2, 2)

            # Convolution Layer
            conv3_1 = tf.nn.relu(self.batch_norm(self.conv2d(conv2_pool, self.weights['wc3_1']) + self.biases['bc3_1']))
            conv3_2 = tf.nn.relu(self.batch_norm(self.conv2d(conv3_1, self.weights['wc3_2']) + self.biases['bc3_2']))
            conv3_3 = tf.nn.relu(self.batch_norm(self.conv2d(conv3_2, self.weights['wc3_3']) + self.biases['bc3_3']))

            # Max Pooling (down-sampling)
            #conv2 = self.maxpool2d(conv2, k=2)
            conv3_pool = self.max_pool(conv3_3, 2, 2)


            # Convolution Layer
            conv4_1 = tf.nn.relu(self.batch_norm(self.conv2d(conv3_pool, self.weights['wc4_1']) + self.biases['bc4_1']))
            conv4_2 = tf.nn.relu(self.batch_norm(self.conv2d(conv4_1, self.weights['wc4_2']) + self.biases['bc4_2']))
            conv4_3 = tf.nn.relu(self.batch_norm(self.conv2d(conv4_2, self.weights['wc4_3']) + self.biases['bc4_3']))

            # Max Pooling (down-sampling)
            #conv2 = self.maxpool2d(conv2, k=2)
            conv4_pool = self.max_pool(conv4_3, 2, 2)

            # Convolution Layer
            conv5_1 = tf.nn.relu(self.batch_norm(self.conv2d(conv4_pool, self.weights['wc5_1']) + self.biases['bc5_1']))
            conv5_2 = tf.nn.relu(self.batch_norm(self.conv2d(conv5_1, self.weights['wc5_2']) + self.biases['bc5_2']))
            conv5_3 = tf.nn.relu(self.batch_norm(self.conv2d(conv5_2, self.weights['wc5_3']) + self.biases['bc5_3']))

            # Max Pooling (down-sampling)
            #conv2 = self.maxpool2d(conv2, k=2)
            #conv5_pool = self.max_pool(conv5_3, 2, 2)



        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        conv5 = tf.reshape(conv5_3, [-1, 2 *2 * 512])
        with tf.name_scope("%sfclayer" % (self.model_type)), tf.variable_scope("%sfclayer" % (self.model_type)):
            fc1 = tf.nn.relu(self.batch_norm(tf.matmul(conv5,self.weights['wd1']) + self.biases['bd1']))
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
            l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
            self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=self.Y)) + l2 * self.weight_decay 
            #optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9,use_nesterov=True)

            self.train_op = optimizer.minimize(self.loss_op)

        with tf.name_scope("%ssummarization" % (self.model_type)), tf.variable_scope(
                        "%ssummarization" % (self.model_type)):
            tf.summary.scalar("loss", self.loss_op)
            # Create a summary to monitor accuracy tensor
            tf.summary.scalar("accuracy", self.accuracy)

            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)


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


    def inference(self, test_x, test_y,levels,stddevVar, load_path ):

        with tf.Session() as self.sess:
            ckpt = tf.train.get_checkpoint_state(load_path)
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)



            acc_ckpt = 0.0
            pre_index = 0
            add = 1000
            for it in range(10):
                batch_x = test_x[pre_index:pre_index+add]
                batch_y = test_y[pre_index:pre_index+add]
                pre_index = pre_index + add
                acc_  = self.sess.run(self.accuracy,feed_dict={self.X:batch_x, self.Y:batch_y, self.keep_prob: 1.0, self.train_flag: False, self.softmax_temperature: 1.0 })
                #loss += loss_ / 10.0
                acc_ckpt += acc_ / 10.0
            print("Testing teacher checkpoint accuracy", acc_ckpt)

            def run_inference_w_variation(test_x, test_y,level,stddevVar):
                rramTensors = [v for v in tf.trainable_variables()]
                #quantize levels
                if level == 0:
                    print("No quantization, Testing in Full Precision")
                else: 
                    quantize(rramTensors, levels = level)
                #quantize(rramTensors, levels = level)
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
                acc = 0.0
                pre_index = 0
                add = 1000
                for it in range(10):
                    batch_x = test_x[pre_index:pre_index+add]
                    batch_y = test_y[pre_index:pre_index+add]
                    pre_index = pre_index + add
                    acc_  = self.sess.run(self.accuracy,feed_dict={self.X:batch_x, self.Y:batch_y, self.keep_prob: 1.0, self.train_flag: False, self.softmax_temperature: 1.0 })
                    #loss += loss_ / 10.0
                    acc += acc_ / 10.0
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
            acc = run_inference_w_variation(test_x, test_y,levels,stddevVar=stddevVar)
            return acc





    def train(self, train_x, train_y, test_x, test_y):
        
        #Learning rate scheduler
        def learning_rate_schedule(epoch_num):
              if epoch_num < 81:
                  return 0.1
              elif epoch_num < 121:
                  return 0.01
              else:
                  return 0.001        

        with tf.Session() as self.sess:

            # Initialize the variables (i.e. assign their default value)
            self.sess.run(tf.global_variables_initializer())

            print("Starting Training")

            train_summary_writer = tf.summary.FileWriter(self.log_dir, graph=self.sess.graph)
            max_accuracy = 0

            def run_testing(ep,test_x,test_y):
                acc = 0.0
                loss = 0.0
                pre_index = 0
                add = 1000
                for it in range(10):
                    batch_x = test_x[pre_index:pre_index+add]
                    batch_y = test_y[pre_index:pre_index+add]
                    pre_index = pre_index + add
                    loss_, acc_  = self.sess.run([self.loss_op,self.accuracy],feed_dict={self.X:batch_x, self.Y:batch_y, self.keep_prob: 1.0, self.train_flag: False, self.softmax_temperature: self.temperature })
                    loss += loss_ / 10.0
                    acc += acc_ / 10.0
                return acc, loss

            def _random_crop(batch, crop_shape, padding=None):
                    oshape = np.shape(batch[0])
                    
                    if padding:
                        oshape = (oshape[0] + 2*padding, oshape[1] + 2*padding)
                    new_batch = []
                    npad = ((padding, padding), (padding, padding), (0, 0))
                    for i in range(len(batch)):
                        new_batch.append(batch[i])
                        if padding:
                            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                                      mode='constant', constant_values=0)
                        nh = random.randint(0, oshape[0] - crop_shape[0])
                        nw = random.randint(0, oshape[1] - crop_shape[1])
                        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                                                    nw:nw + crop_shape[1]]
                    return new_batch

            def _random_flip_leftright(batch):
                    for i in range(len(batch)):
                        if bool(random.getrandbits(1)):
                            batch[i] = np.fliplr(batch[i])
                    return batch


            def data_augmentation(batch):
                batch = _random_flip_leftright(batch)
                batch = _random_crop(batch, [32,32], 4)
                return batch

            for ep in range(1,self.total_epoch+1):
                lr = learning_rate_schedule(ep)
                pre_index = 0
                train_acc = 0.0
                train_loss = 0.0
                start_time = time.time()
                test_acc = []

                print("\nepoch %d/%d:" %(ep,self.total_epoch))

                for it in range(1,self.iterations+1):
                    batch_x = train_x[pre_index:pre_index+self.batch_size]
                    batch_y = train_y[pre_index:pre_index+self.batch_size]

                    batch_x = data_augmentation(batch_x)

                    _, batch_loss = self.sess.run([self.train_op, self.loss_op],feed_dict={self.X:batch_x, self.Y:batch_y, self.keep_prob: self.dropoutprob, self.learning_rate: lr,self.softmax_temperature: self.temperature,self.train_flag: True})
                    batch_acc = self.accuracy.eval(feed_dict={self.X:batch_x, self.Y:batch_y, self.keep_prob: 1.0, self.softmax_temperature: self.temperature, self.train_flag: True})

                    train_loss += batch_loss
                    train_acc  += batch_acc
                    pre_index  += self.batch_size

                    if it == self.iterations:
                        train_loss /= self.iterations
                        train_acc /= self.iterations

                        loss_, acc_  = self.sess.run([self.loss_op,self.accuracy],feed_dict={self.X:batch_x, self.Y:batch_y, self.keep_prob: 1.0, self.softmax_temperature: self.temperature, self.train_flag: True})

                        val_acc, val_loss = run_testing(ep,test_x,test_y)
                        test_acc.append(val_acc)


                        print("iteration: %d/%d, cost_time: %ds, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f" %(it, self.iterations, int(time.time()-start_time), train_loss, train_acc, val_loss, val_acc))
                    else:
                        print("iteration: %d/%d, train_loss: %.4f, train_acc: %.4f" %(it, self.iterations, train_loss / it, train_acc / it) , end='\r')

            save_path = self.saver.save(self.sess, self.checkpoint_path)
            print("Model Checkpointed to %s " % (save_path))
            np.savetxt("cifar10_VGG16_acc.csv", test_acc, delimiter=",")
            train_summary_writer.close()


            print("Optimization Finished!")


    def predict(self, data_X, temperature=1.0):
        return self.sess.run(self.prediction,
                             feed_dict={self.X: data_X, self.keep_prob: 1.0, self.train_flag: False, self.softmax_temperature: temperature})


    def run_inference(self,test_x,test_y):
        acc = 0.0
        loss = 0.0
        pre_index = 0
        add = 1000
        for it in range(10):
            batch_x = test_x[pre_index:pre_index+add]
            batch_y = test_y[pre_index:pre_index+add]
            pre_index = pre_index + add
            loss_, acc_  = self.sess.run([self.loss_op,self.accuracy],feed_dict={self.X:batch_x, self.Y:batch_y, self.keep_prob: 1.0, self.train_flag: False, self.softmax_temperature: self.temperature })
            loss += loss_ / 10.0
            acc += acc_ / 10.0
        print("Testing accuracy", acc)



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
        #self.learning_rate = 0.001
        self.num_steps = args.num_steps
        self.batch_size = args.batch_size
        self.total_epoch = args.total_epoch
        self.display_step = args.display_step
        self.num_input = 784  # MNIST data input (img shape: 28*28)
        self.image_size = 32
        self.num_classes = 10
        self.temperature = args.temperature
        self.checkpoint_dir = args.checkpoint_dir
        self.checkpoint_file = "smallmodel_CIFAR10"
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_file)
        self.max_checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_file + "max")
        self.log_dir = os.path.join(args.log_dir, self.checkpoint_file)
        self.model_type = model_type
        self.iterations = args.iterations
        self.momentum_rate = 0.9
        self.weight_decay = 0.0003


        # Store layers weight & bias
        self.weights = {
            # 5x5 conv, 1 input, 32 outputs
            #'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32]), name="%s_%s" % (self.model_type, "wc1")),
            'wc1_1': tf.get_variable(shape = [3, 3, 3, 64], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc1_1")),
            'wc1_2': tf.get_variable(shape = [3, 3, 64, 64], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc1_2")),

            # 5x5 conv, 32 inputs, 64 outputs
            #'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64]), name="%s_%s" % (self.model_type, "wc2")),
            'wc2_1': tf.get_variable(shape = [3, 3, 64, 128], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc2_1")),
            'wc2_2': tf.get_variable(shape = [3, 3, 128, 128], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc2_2")),


            'wc3_1': tf.get_variable(shape = [3, 3, 128, 256], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc3_1")),
            'wc3_2': tf.get_variable(shape = [3, 3, 256, 256], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc3_2")),
            'wc3_3': tf.get_variable(shape = [3, 3, 256, 256], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc3_3")),


            'wc4_1': tf.get_variable(shape = [3, 3, 256, 512], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc4_1")),
            'wc4_2': tf.get_variable(shape = [3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc4_2")),
            'wc4_3': tf.get_variable(shape = [3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc4_3")),

            'wc5_1': tf.get_variable(shape = [3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc5_1")),
            'wc5_2': tf.get_variable(shape = [3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc5_2")),
            'wc5_3': tf.get_variable(shape = [3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wc5_3")),


            # fully connected, 7*7*64 inputs, 1024 outputs
            #'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024]), name="%s_%s" % (self.model_type, "wd1")),
            'wd1': tf.get_variable(shape = [2 * 2 * 512, 4096], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wd1")),

            # fully connected, 7*7*64 inputs, 1024 outputs
            #'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024]), name="%s_%s" % (self.model_type, "wd1")),
            'wd2': tf.get_variable(shape = [4096, 4096], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "wd2")),

            # 1024 inputs, 10 outputs (class prediction)
            #'out': tf.Variable(tf.random_normal([1024, self.num_classes]), name="%s_%s" % (self.model_type, "out"))
            'out': tf.get_variable(shape = [4096, self.num_classes], initializer=tf.contrib.keras.initializers.he_normal(), name="%s_%s" % (self.model_type, "out"))

        }

        self.biases = {
            'bc1_1': tf.Variable(tf.random_normal([64]), name="%s_%s" % (self.model_type, "bc1_1")),
            'bc1_2': tf.Variable(tf.random_normal([64]), name="%s_%s" % (self.model_type, "bc1_2")),

            'bc2_1': tf.Variable(tf.random_normal([128]), name="%s_%s" % (self.model_type, "bc2_1")),
            'bc2_2': tf.Variable(tf.random_normal([128]), name="%s_%s" % (self.model_type, "bc2_2")),

            'bc3_1': tf.Variable(tf.random_normal([256]), name="%s_%s" % (self.model_type, "bc3_1")),
            'bc3_2': tf.Variable(tf.random_normal([256]), name="%s_%s" % (self.model_type, "bc3_2")),
            'bc3_3': tf.Variable(tf.random_normal([256]), name="%s_%s" % (self.model_type, "bc3_3")),


            'bc4_1': tf.Variable(tf.random_normal([512]), name="%s_%s" % (self.model_type, "bc4_1")),
            'bc4_2': tf.Variable(tf.random_normal([512]), name="%s_%s" % (self.model_type, "bc4_2")),
            'bc4_3': tf.Variable(tf.random_normal([512]), name="%s_%s" % (self.model_type, "bc4_3")),

            'bc5_1': tf.Variable(tf.random_normal([512]), name="%s_%s" % (self.model_type, "bc5_1")),
            'bc5_2': tf.Variable(tf.random_normal([512]), name="%s_%s" % (self.model_type, "bc5_2")),
            'bc5_3': tf.Variable(tf.random_normal([512]), name="%s_%s" % (self.model_type, "bc5_3")),




            'bd1': tf.Variable(tf.random_normal([4096]), name="%s_%s" % (self.model_type, "bd1")),
            'bd2': tf.Variable(tf.random_normal([4096]), name="%s_%s" % (self.model_type, "bd2")),
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
        #self.X = tf.placeholder(tf.float32, [None, self.num_input], name="%s_%s" % (self.model_type, "xinput"))
        self.X = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name="%s_%s" % (self.model_type, "xinput"))
        self.Y = tf.placeholder(tf.float32, [None, self.num_classes], name="%s_%s" % (self.model_type, "yinput"))
        self.keep_prob = tf.placeholder(tf.float32,
                                        name="%s_%s" % (self.model_type, "dropoutprob"))  # dropout (keep probability)
        self.flag = tf.placeholder(tf.bool, None, name="%s_%s" % (self.model_type, "flag"))
        self.soft_Y = tf.placeholder(tf.float32, [None, self.num_classes], name="%s_%s" % (self.model_type, "softy"))
        self.softmax_temperature = tf.placeholder(tf.float32, name="%s_%s" % (self.model_type, "softmaxtemperature"))
        self.train_flag = tf.placeholder(tf.bool)
        self.learning_rate = tf.placeholder(tf.float32)



        # Convolution Layer
        with tf.name_scope("%sconvmaxpool" % (self.model_type)), tf.variable_scope("%sconvmaxpool" % (self.model_type)):
            #conv1 = self.conv2d(x, self.weights['wc1'], self.biases['bc1'])
            conv1_1 = tf.nn.relu(self.batch_norm(self.conv2d(self.X, self.weights['wc1_1']) + self.biases['bc1_1']))
            conv1_2 = tf.nn.relu(self.batch_norm(self.conv2d(conv1_1, self.weights['wc1_2']) + self.biases['bc1_2']))

            # Max Pooling (down-sampling)
            #conv1 = self.maxpool2d(conv1, k=2)
            conv1_pool = self.max_pool(conv1_2, 2, 2)

            # Convolution Layer
            #conv2 = self.conv2d(conv1, self.weights['wc2'], self.biases['bc2'])
            conv2_1 = tf.nn.relu(self.batch_norm(self.conv2d(conv1_pool, self.weights['wc2_1']) + self.biases['bc2_1']))
            conv2_2 = tf.nn.relu(self.batch_norm(self.conv2d(conv2_1, self.weights['wc2_2']) + self.biases['bc2_2']))

            # Max Pooling (down-sampling)
            #conv2 = self.maxpool2d(conv2, k=2)
            conv2_pool = self.max_pool(conv2_2, 2, 2)

            # Convolution Layer
            conv3_1 = tf.nn.relu(self.batch_norm(self.conv2d(conv2_pool, self.weights['wc3_1']) + self.biases['bc3_1']))
            conv3_2 = tf.nn.relu(self.batch_norm(self.conv2d(conv3_1, self.weights['wc3_2']) + self.biases['bc3_2']))
            conv3_3 = tf.nn.relu(self.batch_norm(self.conv2d(conv3_2, self.weights['wc3_3']) + self.biases['bc3_3']))

            # Max Pooling (down-sampling)
            #conv2 = self.maxpool2d(conv2, k=2)
            conv3_pool = self.max_pool(conv3_3, 2, 2)


            # Convolution Layer
            conv4_1 = tf.nn.relu(self.batch_norm(self.conv2d(conv3_pool, self.weights['wc4_1']) + self.biases['bc4_1']))
            conv4_2 = tf.nn.relu(self.batch_norm(self.conv2d(conv4_1, self.weights['wc4_2']) + self.biases['bc4_2']))
            conv4_3 = tf.nn.relu(self.batch_norm(self.conv2d(conv4_2, self.weights['wc4_3']) + self.biases['bc4_3']))

            # Max Pooling (down-sampling)
            #conv2 = self.maxpool2d(conv2, k=2)
            conv4_pool = self.max_pool(conv4_3, 2, 2)

            # Convolution Layer
            conv5_1 = tf.nn.relu(self.batch_norm(self.conv2d(conv4_pool, self.weights['wc5_1']) + self.biases['bc5_1']))
            conv5_2 = tf.nn.relu(self.batch_norm(self.conv2d(conv5_1, self.weights['wc5_2']) + self.biases['bc5_2']))
            conv5_3 = tf.nn.relu(self.batch_norm(self.conv2d(conv5_2, self.weights['wc5_3']) + self.biases['bc5_3']))

            # Max Pooling (down-sampling)
            #conv2 = self.maxpool2d(conv2, k=2)
            #conv5_pool = self.max_pool(conv5_3, 2, 2)



        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        conv5 = tf.reshape(conv5_3, [-1, 2 *2 * 512])
        with tf.name_scope("%sfclayer" % (self.model_type)), tf.variable_scope("%sfclayer" % (self.model_type)):
            fc1 = tf.nn.relu(self.batch_norm(tf.matmul(conv5,self.weights['wd1']) + self.biases['bd1']))
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
            l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
            

            self.total_loss = self.loss_op_standard + l2 * self.weight_decay

            self.loss_op_soft = tf.cond(self.flag,
                                        true_fn=lambda: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                            logits=logits / self.softmax_temperature, labels=self.soft_Y)),
                                        false_fn=lambda: 0.0)
            lamda = 0.07

            self.total_loss = lamda * self.total_loss + self.loss_op_soft
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            #self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum_rate,use_nesterov=True)
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



    def inference(self, test_x, test_y,levels,stddevVar, load_path ):

        with tf.Session() as self.sess:
            ckpt = tf.train.get_checkpoint_state(load_path)
            #if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

            acc_ckpt =  self.sess.run(self.accuracy, feed_dict={self.X: test_x,
                                                                self.Y: test_y,
                                                                self.train_flag: False,
                                                                self.keep_prob: 1.0,
                                                                self.softmax_temperature: 1.0
                                                                })
            print("Testing teacher checkpoint accuracy", acc_ckpt)

            def run_inference_w_variation(test_x, test_y,level,stddevVar):
                rramTensors = [v for v in tf.trainable_variables()]
                #quantize levels
                if level == 0:
                    print("No quantization, Testing in Full Precision")
                else: 
                    quantize(rramTensors, levels = level)
                #quantize(rramTensors, levels = level)
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
                acc =  self.sess.run(self.accuracy, feed_dict={self.X: test_x,
                                                               self.Y: test_y,
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




            print("Teacher Model accuracy with quantization level and variation %s %s" %(levels, stddevVar))
            acc = run_inference_w_variation(test_x, test_y,levels,stddevVar=stddevVar)
            return acc


    def train(self, dataset, train_x, train_y, test_x, test_y, teacher_model=None):
        teacher_flag = False
        if teacher_model is not None:
            teacher_flag = True


        def learning_rate_schedule(epoch_num):
              if epoch_num <41:
                  return 0.01
              elif epoch_num <121:
                  return 0.001
              else:
                  return 0.001 

        with tf.Session() as self.sess:

        #Learning rate scheduler
            # Initialize the variables (i.e. assign their default value)
            self.sess.run(tf.global_variables_initializer())
            var = tf.trainable_variables()
            grads = tf.gradients(self.loss_op_standard,var)
            print(grads)
            train_data = dataset.get_train_data()
            train_summary_writer = tf.summary.FileWriter(self.log_dir, graph=self.sess.graph)

            max_accuracy = 0

            print("Starting Training")

            def dev_step(ep, test_x, test_y):
                validation_x, validation_y = test_x, test_y
                loss, acc = self.sess.run([self.loss_op_standard, self.accuracy], feed_dict={self.X: validation_x,
                                                                                             self.Y: validation_y,
                                                                                             # self.soft_Y: validation_y,
                                                                                             self.keep_prob: 1.0,
                                                                                             self.flag: False,
                                                                                             self.train_flag: False,
                                                                                             self.softmax_temperature: 1.0})

                #if acc > max_accuracy:
                #    save_path = self.saver.save(self.sess, self.checkpoint_path)
                #    print("Model Checkpointed to %s " % (save_path))

                print("Epoch " + str(ep) + ", Validation Loss= " + "{:.4f}".format(
                    loss) + ", Validation Accuracy= " + "{:.3f}".format(acc))


            def run_testing(ep,test_x,test_y):
                acc = 0.0
                loss = 0.0
                pre_index = 0
                add = 1000
                for it in range(10):
                    batch_x = test_x[pre_index:pre_index+add]
                    batch_y = test_y[pre_index:pre_index+add]
                    pre_index = pre_index + add
                    loss_, acc_  = self.sess.run([self.loss_op_standard,self.accuracy],feed_dict={self.X:batch_x, self.Y:batch_y, self.flag: False, self.keep_prob: 1.0, self.train_flag: False, self.softmax_temperature: 1.0 })
                    loss += loss_ / 10.0
                    acc += acc_ / 10.0
                return acc, loss

            def _random_crop(batch, crop_shape, padding=None):
                    oshape = np.shape(batch[0])
                    
                    if padding:
                        oshape = (oshape[0] + 2*padding, oshape[1] + 2*padding)
                    new_batch = []
                    npad = ((padding, padding), (padding, padding), (0, 0))
                    for i in range(len(batch)):
                        new_batch.append(batch[i])
                        if padding:
                            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                                      mode='constant', constant_values=0)
                        nh = random.randint(0, oshape[0] - crop_shape[0])
                        nw = random.randint(0, oshape[1] - crop_shape[1])
                        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                                                    nw:nw + crop_shape[1]]
                    return new_batch

            def _random_flip_leftright(batch):
                    for i in range(len(batch)):
                        if bool(random.getrandbits(1)):
                            batch[i] = np.fliplr(batch[i])
                    return batch


            def data_augmentation(batch):
                batch = _random_flip_leftright(batch)
                batch = _random_crop(batch, [32,32], 4)
                return batch



            def run_inference_w_variation(test_x,test_y,stddevVar=0.5):
                #test_images, test_labels = dataset.get_test_data()
                rramTensors = [v for v in tf.trainable_variables()]
                #quantize levels
                quantize(rramTensors, levels=10)
                #Add device variations
                if stddevVar != 0.0:
                    allParameters = [v.eval() for v in rramTensors]
                    allShapes = [v.get_shape().as_list() for v in rramTensors]
                    for i in range(len(allParameters)):
                        if ('beta' not in rramTensors[i].name) or ('gamma' not in rramTensors[i].name):
                            print("rramTensors name", rramTensors[i].name)
                            param = allParameters[i]*np.exp(np.random.normal(0, 0.3, allShapes[i]))
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
                print("Testing Accuracy with variation:", self.sess.run(self.accuracy, feed_dict={self.X: test_x,
                                                                                   self.Y: test_y,
                                                                                   self.flag: False,
                                                                                   self.train_flag: False,
                                                                                   self.keep_prob: 1.0,
                                                                                   self.softmax_temperature: 1.0
                                                                                   }))



            def run_testing_w_variations(test_x,test_y,stddevVar= 0.5):
                acc = 0.0
                loss = 0.0
                pre_index = 0
                add = 1000
                rramTensors = [v for v in tf.trainable_variables()]
                #quantize levels
                quantize(rramTensors, levels=10)
                #Add device variations
                if stddevVar != 0.0:
                    allParameters = [v.eval() for v in rramTensors]
                    allShapes = [v.get_shape().as_list() for v in rramTensors]
                    for i in range(len(allParameters)):
                        if ('beta' not in rramTensors[i].name) or ('gamma' not in rramTensors[i].name):
                            print("rramTensors name", rramTensors[i].name)
                            param = allParameters[i]*np.exp(np.random.normal(0, 0.3, allShapes[i]))
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
                ##SF1
                addSA1(rramTensors, cfg.RRAM.SA1)

                for it in range(10):
                    batch_x = test_x[pre_index:pre_index+add]
                    batch_y = test_y[pre_index:pre_index+add]
                    pre_index = pre_index + add
                    loss_, acc_  = self.sess.run([self.loss_op_standard,self.accuracy],feed_dict={self.X:batch_x, self.Y:batch_y,self.flag: False, self.keep_prob: 1.0, self.train_flag: False, self.softmax_temperature: 1.0 })
                    loss += loss_ / 10.0
                    acc += acc_ / 10.0
                print("Testing accuracy", acc)

                #print("Testing Accuracy:", self.sess.run(self.accuracy, feed_dict={self.X: test_x,
                #                                                                   self.Y: test_y,
                #                                                                   self.flag: False,
                #                                                                   self.train_flag: False,
                #                                                                   self.keep_prob: 1.0,
                #                                                                   self.softmax_temperature: 1.0
                #                                                                   }))










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





            def addDeviceVariation(stddevVar=0.5):
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

            print("Adding variations first time")


            for ep in range(1,self.total_epoch+1):
                pre_index = 0
                lr = learning_rate_schedule(ep)
                #if ep %3 == 0:
                print("Adding variations")
                train_vars = tf.trainable_variables()
                quantize(train_vars, levels=10)
                addDeviceVariation(stddevVar=0.5)
                dev_step(ep,test_x, test_y)
                for step in range(1, self.iterations + 1):
                    #batch_x, batch_y = train_data.next_batch(self.batch_size)
                    batch_x = train_x[pre_index:pre_index+self.batch_size]
                    batch_y = train_y[pre_index:pre_index+self.batch_size]
                    batch_x = data_augmentation(batch_x)
                    soft_targets = batch_y
                    if teacher_flag:
                        soft_targets = teacher_model.predict(batch_x, self.temperature)

                    _, summary = self.sess.run([self.train_op, self.merged_summary_op],
                                               feed_dict={self.X: batch_x,
                                                          self.Y: batch_y,
                                                          self.soft_Y: soft_targets,
                                                          self.flag: teacher_flag,
                                                          self.train_flag: True,
                                                          self.keep_prob: 0.5,
                                                          self.learning_rate: lr,
                                                          self.softmax_temperature: self.temperature}
                                               )
                    pre_index  += self.batch_size
                dev_step(ep,test_x, test_y)


            # Final Evaluation and checkpointing before training ends
            dev_step(1, test_x, test_y)
       
            print("Student Model accuracy with variation 0.5")
            run_inference_w_variation(test_x,test_y,stddevVar=0.5)


            train_summary_writer.close()

            print("Optimization Finished!")


    def predict(self, data_X, temperature=1.0):
        return self.sess.run(self.prediction,
                             feed_dict={self.X: data_X, self.flag: False, self.train_flag: False,self.keep_prob: 1.0, self.softmax_temperature: temperature})





    def run_inference(self,test_x,test_y):
        acc = 0.0
        loss = 0.0
        pre_index = 0
        add = 1000
        for it in range(10):
            batch_x = test_x[pre_index:pre_index+add]
            batch_y = test_y[pre_index:pre_index+add]
            pre_index = pre_index + add
            loss_, acc_  = self.sess.run([self.loss_op_standard,self.accuracy],feed_dict={self.X:batch_x, self.Y:batch_y,self.flag: False, self.keep_prob: 1.0, self.train_flag: False, self.softmax_temperature: 1.0 })
            loss += loss_ / 10.0
            acc += acc_ / 10.0
        print("Testing accuracy", acc)


    def addDeviceVariation(self, rramTensors, stddevVar=0.5):
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
