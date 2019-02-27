import argparse
import tensorflow as tf
import numpy as np
import os
import sys
import importlib
import json
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))

sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util
import pickle

# Execution
# python test_n_est_w_experts.py --results_path='log/my_experts_kinect/' --model='experts_n_est' --dataset_name='' --dataset_path='/home/itzik/Datasets/nyu v2/nyu_v2_txt/' --testset='testset.txt' --sparse_patches=0 --batch_size=128
parser = argparse.ArgumentParser()
parser.add_argument('--results_path', default='log/my_experts/', help='path to trained model, default log/my_experts/')
parser.add_argument('--model', default='experts_n_est', help='Model name [default: ms_norm_est]')
parser.add_argument('--dataset_name', type=str, default='pcpnet', help='Relative path to data directory, default pcpnet')
parser.add_argument('--dataset_path', type=str, default=None, help='full path to dataset for datasets outside the local data dir')
parser.add_argument('--sparse_patches', type=int, default=False,
                    help='test on a subset of thepoints in each point cloud in the test data, default False')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch_size', type=int, default=128, help='Batch Size during training [default: 128]')
parser.add_argument('--testset', type=str, default='testset_temp.txt', help='test set file name, default testset_temp.txt')
FLAGS = parser.parse_args()

# DEFAULT SETTINGS
results_path = FLAGS.results_path
pretrained_model_path = results_path + 'model.ckpt'
model_str = FLAGS.model

if FLAGS.dataset_path is None:
    PC_PATH = os.path.join(BASE_DIR, 'data/' + FLAGS.dataset_name + '/')
else:
    PC_PATH = FLAGS.dataset_path

TEST_FILES = PC_PATH + FLAGS.testset
SPARSE_PATCHES = FLAGS.sparse_patches
BATCH_SIZE = FLAGS.batch_size
GPU_IDX = FLAGS.gpu

params = pickle.load(open(results_path + 'parameters.p', "rb"))  # load training paramters
PATCH_RADIUS = params.patch_radius
n_rad = len(PATCH_RADIUS)
EXPERT_LOSS_TYPE = params.expert_loss_type
LOSS_TYPE = params.loss_type
N_EXPERTS = params.n_experts
NUM_POINT = params.num_point  # the max number of points in the all testing data shapes
EXPERT_DICT = json.loads(params.expert_dict)
EXPERT_DICT = {int(key): json.loads(value.encode('UTF8')) for key, value in EXPERT_DICT.iteritems()}

output_dir = os.path.join(results_path, FLAGS.dataset_name + '_results/')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

sys.path.append(os.path.join(results_path))
MODEL = importlib.import_module(model_str)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# MAIN SCRIPT
def printout(flog, data):
    print(data)
    flog.write(data + '\n')
    sys.stdout.flush()

def predict(gmm):

    with tf.device('/gpu:' + str(GPU_IDX)):
        points_pl, normal_pl, w_pl, mu_pl, sigma_pl, n_effective_points = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, gmm, PATCH_RADIUS)

        is_training_pl = tf.placeholder(tf.bool, shape=())

        # Get model and loss
        experts_prob, n_pred, fv = MODEL.get_model(points_pl, w_pl, mu_pl, sigma_pl, is_training_pl, PATCH_RADIUS,
                                                   original_n_points=n_effective_points, n_experts=N_EXPERTS, expert_dict=EXPERT_DICT)
        loss, cos_ang = MODEL.get_loss(n_pred, normal_pl, experts_prob, loss_type=LOSS_TYPE, n_experts=N_EXPERTS,
                                           expert_type=EXPERT_LOSS_TYPE)
        tf.summary.scalar('loss', loss)
        ops = {'points_pl': points_pl,
               'normal_pl': normal_pl,
               'n_effective_points': n_effective_points,
               'experts_prob': experts_prob,
               'cos_ang': cos_ang,
               'w_pl': w_pl,
               'mu_pl': mu_pl,
               'sigma_pl': sigma_pl,
               'is_training_pl': is_training_pl,
               'fv': fv,
               'n_pred': n_pred,
               'loss': loss
               }

    saver = tf.train.Saver()
    sess = tf_util.get_session(GPU_IDX, limit_gpu=True)

    flog = open(os.path.join(output_dir, 'log.txt'), 'w')

    # Restore model variables from disk.
    printout(flog, 'Loading model %s' % pretrained_model_path)
    saver.restore(sess, pretrained_model_path)
    printout(flog, 'Model restored.')

    # PCPNet data loaders
    testnset_loader, dataset = provider.get_data_loader(dataset_name=TEST_FILES, batchSize=BATCH_SIZE, indir=PC_PATH,
                                             patch_radius=PATCH_RADIUS,
                                             points_per_patch=NUM_POINT, outputs=[],
                                             patch_point_count_std=0,
                                             seed=3627473, identical_epochs=False, use_pca=False, patch_center='point',
                                             point_tuple=1, cache_capacity=100,
                                             patch_sample_order='full',
                                             workers=0, dataset_type='test', sparse_patches=SPARSE_PATCHES)

    is_training = False

    shape_ind = 0
    shape_patch_offset = 0
    shape_patch_count = dataset.shape_patch_count[shape_ind]
    normal_prop = np.zeros([shape_patch_count, 3])
    expert_prop = np.zeros([shape_patch_count, ],  dtype=np.uint64)
    expert_prob_props = np.zeros([shape_patch_count, N_EXPERTS])
    num_batchs = len(testnset_loader)


    for batch_idx, data in enumerate(testnset_loader, 0):

        current_data = data[0]
        n_effective_points = data[-1]

        if current_data.shape[0] < BATCH_SIZE:
            # compensate for last batch
            pad_size = current_data.shape[0]
            current_data = np.concatenate([current_data,
                                           np.zeros([BATCH_SIZE - pad_size, n_rad*NUM_POINT, 3])], axis=0)
            n_effective_points = np.concatenate([n_effective_points,
                                           np.zeros([BATCH_SIZE - pad_size, n_rad])], axis=0)

        feed_dict = {ops['points_pl']: current_data,
                     ops['n_effective_points']: n_effective_points,
                     ops['w_pl']: gmm.weights_,
                     ops['mu_pl']: gmm.means_,
                     ops['sigma_pl']: np.sqrt(gmm.covariances_),
                     ops['is_training_pl']: is_training, }
        n_est, experts_prob = sess.run([ ops['n_pred'], ops['experts_prob']], feed_dict=feed_dict)

        expert_to_use = np.argmax(experts_prob, axis=0)
        experts_prob = np.transpose(experts_prob)
        n_est = n_est[expert_to_use, range(len(expert_to_use))]

        # Save estimated normals to file
        batch_offset = 0

        print('Processing batch  [%d/%d]...' % (batch_idx, num_batchs-1))

        while batch_offset < n_est.shape[0] and shape_ind + 1 <= len(dataset.shape_names):
            shape_patches_remaining = shape_patch_count - shape_patch_offset
            batch_patches_remaining = n_est.shape[0] - batch_offset

            # append estimated patch properties batch to properties for the current shape on the CPU
            normal_prop[shape_patch_offset:shape_patch_offset + min(shape_patches_remaining,
                                                                          batch_patches_remaining), :] = \
                n_est[batch_offset:batch_offset + min(shape_patches_remaining, batch_patches_remaining), :]

            expert_prop[shape_patch_offset:shape_patch_offset + min(shape_patches_remaining,
                                                                          batch_patches_remaining)] = \
                expert_to_use[batch_offset:batch_offset + min(shape_patches_remaining, batch_patches_remaining)]

            expert_prob_props[shape_patch_offset:shape_patch_offset + min(shape_patches_remaining,
                                                                          batch_patches_remaining), :] = \
                experts_prob[batch_offset:batch_offset + min(shape_patches_remaining, batch_patches_remaining), :]

            batch_offset = batch_offset + min(shape_patches_remaining, batch_patches_remaining)
            shape_patch_offset = shape_patch_offset + min(shape_patches_remaining, batch_patches_remaining)


            if shape_patches_remaining <= batch_patches_remaining:

                np.savetxt(os.path.join(output_dir, dataset.shape_names[shape_ind] + '.normals'),
                           normal_prop)
                print('saved normals for ' + dataset.shape_names[shape_ind])
                np.savetxt(os.path.join(output_dir, dataset.shape_names[shape_ind] + '.experts'),
                           expert_prop.astype(int),  fmt='%i')
                np.savetxt(os.path.join(output_dir, dataset.shape_names[shape_ind] + '.experts_probs'),
                           expert_prob_props)
                print('saved experts for ' + dataset.shape_names[shape_ind])
                shape_patch_offset = 0
                shape_ind += 1
                if shape_ind < len(dataset.shape_names):
                    shape_patch_count = dataset.shape_patch_count[shape_ind]
                    normal_prop = np.zeros([shape_patch_count, 3])
                    expert_prop = np.zeros([shape_patch_count, ], dtype=np.uint64)
                    expert_prob_props = np.zeros([shape_patch_count, N_EXPERTS])
                sys.stdout.flush()


with tf.Graph().as_default():
    gmm = pickle.load(open(results_path+'gmm.p', "rb"))
    predict(gmm)
