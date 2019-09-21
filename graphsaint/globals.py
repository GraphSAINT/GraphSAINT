import tensorflow as tf
import numpy as np
import os,sys,time,datetime
from os.path import expanduser
import pdb


import subprocess
git_rev = subprocess.Popen("git rev-parse --short HEAD", shell=True, stdout=subprocess.PIPE, universal_newlines=True).communicate()[0]
git_branch = subprocess.Popen("git symbolic-ref --short -q HEAD", shell=True, stdout=subprocess.PIPE, universal_newlines=True).communicate()[0]

timestamp = time.time()
timestamp = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H:%M:%S')




# Set random seed
#seed = 123
#np.random.seed(seed)
#tf.set_random_seed(seed)


flags = tf.app.flags
FLAGS = flags.FLAGS
# Settings
flags.DEFINE_integer('num_cpu_core', 20, 'Number of CPU cores for parallel sampling')
flags.DEFINE_boolean('log_device_placement', False, "Whether to log device placement.")
flags.DEFINE_string('data_prefix', '', 'prefix identifying training data. must be specified.')
flags.DEFINE_string('log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_string('gpu','-1234', "which gpu to use.")
flags.DEFINE_integer('eval_train_every', 15, "How often to evaluate training subgraph accuracy.")

flags.DEFINE_string('train_config', '', "path to the configuration of training (*.yml)")
flags.DEFINE_string('model','','pretrained model')
flags.DEFINE_string('dtype','s','d for double, s for single precision floating point')
flags.DEFINE_boolean('timeline',False,'to save timeline.json or not')
flags.DEFINE_boolean('tensorboard',False,'to save data to tensorboard or not')
flags.DEFINE_boolean('logging',False,'log input and output histogram of each layer')
flags.DEFINE_boolean('dualGPU',False,'whether to distribute the model to two GPU')
flags.DEFINE_boolean('cpu_eval',False,'whether to use CPU to do evaulation')

NUM_PAR_SAMPLER = FLAGS.num_cpu_core
SAMPLES_PER_PROC = -(-(200 // NUM_PAR_SAMPLER)) # round up division


if FLAGS.data_prefix.split('/')[-1]=='amazon':
    FLAGS.cpu_eval=True


# auto choosing available NVIDIA GPU
gpu_selected = FLAGS.gpu
if gpu_selected == '-1234':
    # auto detect gpu by filtering on the nvidia-smi command output
    gpu_stat = subprocess.Popen("nvidia-smi",shell=True,stdout=subprocess.PIPE,universal_newlines=True).communicate()[0]
    gpu_avail = set([str(i) for i in range(8)])
    for line in gpu_stat.split('\n'):
        if 'python' in line:
            if line.split()[1] in gpu_avail:
                gpu_avail.remove(line.split()[1])
            if len(gpu_avail) == 0:
                gpu_selected = -2
            else:
                gpu_selected = sorted(list(gpu_avail))[0]
    if gpu_selected == -1:
        gpu_selected = '0'
if str(gpu_selected).startswith('nvlink'):
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_selected).split('nvlink')[1]
elif int(gpu_selected) >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_selected)
    GPU_MEM_FRACTION = 0.8
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"


# global vars

f_mean = lambda l: sum(l)/len(l)

F_ACT = {'I': lambda x:x,
         'relu': tf.nn.relu,
         'leaky_relu': tf.nn.leaky_relu}

DTYPE = tf.float32 if FLAGS.dtype=='s' else tf.float64
