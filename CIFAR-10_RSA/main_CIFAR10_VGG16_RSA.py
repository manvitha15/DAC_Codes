from tensorflow.examples.tutorials.mnist import input_data
import os
import argparse
import model_CIFAR10_VGG16_RSA
import data
import tensorflow as tf
import numpy as np
import random
from data_utility import *
seed = int(os.getenv("SEED", 12))
tf.set_random_seed(seed)
np.random.seed(seed)
random.seed(seed)


def check_and_makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def convert_str_to_bool(text):
    if text.lower() in ["true", "yes", "y", "1"]:
        return True
    else:
        return False


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--display_step', type=int, default=100)
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoint")
    parser.add_argument('--log_dir', type=str, default="logs")
    parser.add_argument('--gpu', type=int, default=None, choices=[None, 0, 1])

    # Training Parameters
    parser.add_argument('--load_teacher_from_checkpoint', type=str, default="false")
    parser.add_argument('--load_teacher_checkpoint_dir', type=str, default=None)
    parser.add_argument('--model_type', type=str, default="teacher", choices=["teacher", "student"])
    parser.add_argument('--num_steps', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=250)
    parser.add_argument('--total_epoch', type=int, default=60)
    parser.add_argument('--iterations', type=int, default=200)


    # Model Parameters
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--dropoutprob', type=float, default=0.75)

    return parser


def setup(args):
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % (args.gpu)

    args.load_teacher_from_checkpoint = convert_str_to_bool(args.load_teacher_from_checkpoint)

    check_and_makedir(args.log_dir)
    check_and_makedir(args.checkpoint_dir)


def main():
    parser = get_parser()
    args = parser.parse_args()
    setup(args)
    mnist = input_data.read_data_sets("./", one_hot=True)
    dataset = data.Dataset(args)

    #CIFAR10 dataset
    def data_preprocessing(x_train,x_test):
    
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
    
        x_train[:,:,:,0] = (x_train[:,:,:,0] - np.mean(x_train[:,:,:,0])) / np.std(x_train[:,:,:,0])
        x_train[:,:,:,1] = (x_train[:,:,:,1] - np.mean(x_train[:,:,:,1])) / np.std(x_train[:,:,:,1])
        x_train[:,:,:,2] = (x_train[:,:,:,2] - np.mean(x_train[:,:,:,2])) / np.std(x_train[:,:,:,2])
    
        x_test[:,:,:,0] = (x_test[:,:,:,0] - np.mean(x_test[:,:,:,0])) / np.std(x_test[:,:,:,0])
        x_test[:,:,:,1] = (x_test[:,:,:,1] - np.mean(x_test[:,:,:,1])) / np.std(x_test[:,:,:,1])
        x_test[:,:,:,2] = (x_test[:,:,:,2] - np.mean(x_test[:,:,:,2])) / np.std(x_test[:,:,:,2])
    
        return x_train, x_test



    train_x, train_y, test_x, test_y = prepare_data()
    train_x, test_x = data_preprocessing(train_x, test_x)
    


    test_acc = []
    tf.reset_default_graph()
    if args.model_type == "student":
        teacher_model = None
        if args.load_teacher_from_checkpoint:
            teacher_model = model_CIFAR10_VGG16_RSA.BigModel(args, "teacher")
            teacher_model.start_session()
            teacher_model.load_model_from_file(args.load_teacher_checkpoint_dir)
            print("Verify Teacher State before Training Student")
            teacher_model.run_inference(test_x, test_y)
        student_model = model_CIFAR10_VGG16_RSA.SmallModel(args, "student")

        ## Testing student model on the best model based on validation set
        student_model.start_session()
        student_model.load_model_from_file(args.checkpoint_dir)
        quantize_levels = 10

        acc = student_model.inference_RSA(train_x, train_y,test_x, test_y,quantize_levels,0.5,5,teacher_model,args.checkpoint_dir)
        print("Final accuracy is", acc)

        if args.load_teacher_from_checkpoint:
            print("Verify Teacher State After Training student Model")
            teacher_model.run_inference(test_x, test_y)
            teacher_model.close_session()
        student_model.close_session()
    else:

        teacher_model = model_CIFAR10_VGG16_RSA.BigModel(args, "teacher")
        ##teacher_model.start_session()
        teacher_model.train(train_x, train_y, test_x, test_y)

        teacher_model.start_session()
        # Testing teacher model on the best model based on validation set
        teacher_model.load_model_from_file(args.checkpoint_dir)
        teacher_model.run_inference(test_x, test_y)
        teacher_model.close_session()   


if __name__ == '__main__':
    main()
    # INVOCATION

    # Teacher
    # python main.py --model_type teacher --checkpoint_dir teachercpt --num_steps 50

    # Student
    # python main.py --model_type student --checkpoint_dir studentcpt --num_steps 50 --gpu 0

    # Student
    # python main.py --model_type student --checkpoint_dir studentcpt --load_teacher_from_checkpoint true --load_teacher_checkpoint_dir teachercpt --num_steps 50
