import os
import sys
import numpy as np
import importlib
import argparse
import tensorflow as tf
import pickle
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tf_util
import provider
import utils
import eulerangles

#Execute
# python train_n_est_w_experts.py   --model='experts_n_est' --gpu=0 --n_experts=7 --expert_loss_type='simple' --expert_dict='{"0": "[0]", "1": "[0]", "2": "[1]", "3": "[1]", "4": "[2]", "5": "[2]", "6": "[0, 1, 2]"}' --insert_rotation_augmentation=0  --log_dir='my_experts' --patch_radius 0.01 0.03 0.05   --loss_type='sin' --batch_size=64 --num_point=512 --identical_epochs=0 --num_gaussians=8 --gmm_variance=0.0156 --learning_rate=0.0001 --max_epoch=1000 --momentum=0.9 --optimizer='adam'  --weight_decay=0.0 --decay_rate=0.7 --decay_step=491520 --trainset='trainingset_whitenoise.txt' --testset='validationset.txt' --desc='run with smaller small scale'

parser = argparse.ArgumentParser()
# Parameters for learning
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--desc', type=str, default='My training run', help='description')
parser.add_argument('--data_path', type=str, default='data/pcpnet/', help='Relative path to data directory')
parser.add_argument('--model', default='experts_n_est', help='Model name [default: 3dmfv_net_cls]')
parser.add_argument('--log_dir', default='my_experts', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=512, help='Neighboring point Number [128/256/512/1024] [default: 256]')
parser.add_argument('--max_epoch', type=int, default=1000, help='Epoch to run [default: 200]')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 64]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=8*1024*15, help='Decay step for lr decay [default: 1024*8*15]') #1024 patches per shape x number of shapes x 15 epochs
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay coef [default: 0.0]')
parser.add_argument('--identical_epochs', type=int, default=False,
                    help='use same patches in each epoch, mainly for debugging')
parser.add_argument('--loss_type', type=str, default='sin', help='loss type [euclidean]')
parser.add_argument('--outputs', type=str, nargs='+', default=['unoriented_normals'],
                    help='outputs of the network, a list with elements of:\n'
                         'unoriented_normals: unoriented (flip-invariant) point normals\n'
                         'oriented_normals: oriented point normals\n'
                         'max_curvature: maximum curvature\n'
                         'min_curvature: mininum curvature\n'
                         'noise: shape noise')
parser.add_argument('--patch_radius', type=float, default=[0.005, 0.01, 0.03], nargs='+', help='patch radius'
                            ' in multiples of the shape\'s bounding box diagonal, multiple values for multi-scale.')
parser.add_argument('--trainset', type=str, default='trainingset_temp.txt', help='training set file name')
parser.add_argument('--testset', type=str, default='validationset_temp.txt', help='test set file name')
parser.add_argument('--insert_rotation_augmentation', type=int, default=False, help='Insert rotation augmentations during training')

# Parameters for GMM
parser.add_argument('--num_gaussians', type=int, default=3, help='number of gaussians for gmm, [default: 3, i.e. 27 gaussians, for improved performance use 8]')
parser.add_argument('--gmm_variance', type=float,  default=0.111, help='variance for grid gmm, recommended use (1/num_gaussians)^2')

#Parameters for Experts
parser.add_argument('--n_experts', type=int, default=7, help='number of expert models [default: 7]')
parser.add_argument('--expert_loss_type', type=str, default='simple', help='type of expert loss')
parser.add_argument('--expert_dict', type=str,
                    default='{"0": "[0]", "1": "[0]", "2": "[1]", "3": "[1]", "4": "[2]", "5": "[2]", "6": "[0, 1, 2]"}',
                    help=' mapping between experts and radius scales ')
FLAGS = parser.parse_args()

EXPERT_DICT = json.loads(FLAGS.expert_dict)
EXPERT_DICT = {int(key): json.loads(value.encode('UTF8')) for key, value in EXPERT_DICT.iteritems()}

N_GAUSSIANS = FLAGS.num_gaussians
GMM_VARIANCE = FLAGS.gmm_variance

PC_PATH = os.path.join(BASE_DIR, FLAGS.data_path)
PATCH_RADIUS = FLAGS.patch_radius
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
WEIGHT_DECAY = FLAGS.weight_decay
VALIDATION_FILES = PC_PATH + FLAGS.testset
TRAIN_FILES = PC_PATH + FLAGS.trainset
PATCHES_PER_SHAPE = 1024
IDENTICAL_EPOCHS = FLAGS.identical_epochs
OUTPUTS = FLAGS.outputs
LOSS_TYPE = FLAGS.loss_type
N_EXPERTS = FLAGS.n_experts
EXPERT_LOSS_TYPE = FLAGS.expert_loss_type
INSERT_ROTATIONS = FLAGS.insert_rotation_augmentation

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')

#Creat log directory ant prevent over-write by creating numbered subdirectories
LOG_DIR = 'log/' + FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
else:
    print('Log dir already exists! creating a new one..............')
    n = 0
    while True:
        n+=1
        new_log_dir = LOG_DIR+'/'+str(n)
        if not os.path.exists(new_log_dir):
            os.makedirs(new_log_dir)
            print('New log dir:'+new_log_dir)
            break
    FLAGS.log_dir = new_log_dir
    LOG_DIR = new_log_dir

desc_filename = os.path.join(LOG_DIR, 'description.txt')
# save description
with open(desc_filename, 'w+') as text_file:
    text_file.write(FLAGS.desc+'\n')
    text_file.flush()

os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train_n_est_w_experts.py %s' % (LOG_DIR)) # bkp of train procedure
pickle.dump(FLAGS, open( os.path.join(LOG_DIR, 'parameters.p'), "wb" ) )

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

LIMIT_GPU = True

# colored console output
green = lambda x: '\033[92m' + x + '\033[0m'
blue = lambda x: '\033[94m' + x + '\033[0m'
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.000001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train(gmm):

    # Build Graph, train and classify
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):

            points_pl, normal_pl, w_pl, mu_pl, sigma_pl, n_effective_points = \
                MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, gmm, PATCH_RADIUS)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter that tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            experts_prob, n_pred, fv = MODEL.get_model(points_pl, w_pl, mu_pl, sigma_pl, is_training_pl, PATCH_RADIUS,
                                                       original_n_points=n_effective_points, bn_decay=bn_decay,
                                                       weight_decay=WEIGHT_DECAY, n_experts=N_EXPERTS, expert_dict=EXPERT_DICT)
            loss, cos_ang = MODEL.get_loss(n_pred, normal_pl, experts_prob, loss_type=LOSS_TYPE, n_experts=N_EXPERTS,
                                           expert_type=EXPERT_LOSS_TYPE)
            tf.summary.scalar('loss', loss)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)#, aggregation_method = tf.AggregationMethod.EXPERIMENTAL_TREE) #consider using: tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        sess = tf_util.get_session(GPU_INDEX, limit_gpu=LIMIT_GPU)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl: True})

        ops = {'points_pl': points_pl,
               'normal_gt_pl': normal_pl,
               'experts_prob': experts_prob,
               'normal_pred': n_pred,
               'n_effective_points': n_effective_points,
               'w_pl': w_pl,
               'mu_pl': mu_pl,
               'sigma_pl': sigma_pl,
               'is_training_pl': is_training_pl,
               'fv': fv,
               'loss': loss,
               'cos_ang': cos_ang,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        trainset, _ = provider.get_data_loader(dataset_name=TRAIN_FILES, batchSize=BATCH_SIZE, indir=PC_PATH, patch_radius=PATCH_RADIUS,
                        points_per_patch=NUM_POINT, outputs=OUTPUTS, patch_point_count_std=0,
                        seed=3627473, identical_epochs=IDENTICAL_EPOCHS, use_pca=False, patch_center='point',
                        point_tuple=1, cache_capacity=100, patches_per_shape=PATCHES_PER_SHAPE, patch_sample_order='random',
                        workers=0, dataset_type='training')
        validationset, validation_dataset = provider.get_data_loader(dataset_name=VALIDATION_FILES, batchSize=BATCH_SIZE, indir=PC_PATH, patch_radius=PATCH_RADIUS,
                        points_per_patch=NUM_POINT, outputs=OUTPUTS, patch_point_count_std=0,
                        seed=3627473, identical_epochs=IDENTICAL_EPOCHS, use_pca=False, patch_center='point',
                        point_tuple=1, cache_capacity=100, patches_per_shape=PATCHES_PER_SHAPE, patch_sample_order='random',
                        workers=0, dataset_type='validation')

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, gmm, train_writer, trainset, epoch)
            eval_one_epoch(sess, ops, gmm, test_writer, validationset, validation_dataset)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, gmm, train_writer, trainset_loader, epoch_num):
    is_training = True

    train_enum = enumerate(trainset_loader, 0)
    train_num_batchs = len(trainset_loader)

    total_seen = 0
    loss_sum = 0

    for batch_idx, data in train_enum:

        current_data = data[0]
        target = tuple(t.data.numpy() for t in data[1:-1])
        current_normals = target[0]
        n_effective_points = np.squeeze(data[-1])
        if INSERT_ROTATIONS:
            angles = 2*np.pi*np.random.randn(3)
            R = np.transpose(eulerangles.euler2mat(z=angles[0], y=angles[1], x=angles[2]))
            rotated_data = np.zeros(current_data.shape, dtype=np.float32)
            rotated_normals = np.zeros(current_normals.shape, dtype=np.float32)
            for k in xrange(current_data.shape[0]):
                shape_pc = current_data[k, ...]
                normal_pc = current_normals[k, ...]
                rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
                rotated_normals[k, ...] = np.dot(normal_pc, R)
            current_data = rotated_data
            current_normals = rotated_normals

        feed_dict = {ops['points_pl']: current_data,
                     ops['normal_gt_pl']: current_normals,
                     ops['n_effective_points']: n_effective_points,
                     ops['w_pl']: gmm.weights_,
                     ops['mu_pl']: gmm.means_,
                     ops['sigma_pl']: np.sqrt(gmm.covariances_),
                     ops['is_training_pl']: is_training, }

        summary, step, _, loss_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss']],
                                                        feed_dict=feed_dict)

        train_writer.add_summary(summary, step)
        total_seen += BATCH_SIZE
        loss_sum += loss_val
        print('epoch %d, [%d/%d] %s loss: %f' % (epoch_num, batch_idx, train_num_batchs - 1, green('train'), loss_val))

    log_string('mean loss: %f' % (loss_sum / float(train_num_batchs)))


def eval_one_epoch(sess, ops, gmm, test_writer, testset_loader, dataset):

    is_training = False
    loss_sum = 0
    total_seen = 0
    test_enum = enumerate(testset_loader, 0)
    test_num_batchs = len(testset_loader)
    n_shapes = len(dataset.shape_names)
    ang_err = []

    for batch_idx, data in test_enum:

        current_data = data[0]
        target = tuple(t.data.numpy() for t in data[1:-1])
        current_normals = target[0]
        n_effective_points = np.squeeze(data[-1])
        feed_dict = {ops['points_pl']: current_data,
                     ops['normal_gt_pl']: current_normals,
                     ops['n_effective_points']: n_effective_points,
                     ops['w_pl']: gmm.weights_,
                     ops['mu_pl']: gmm.means_,
                     ops['sigma_pl']: np.sqrt(gmm.covariances_),
                     ops['is_training_pl']: is_training}

        summary, step, loss_val, experts_prob, n_est, cos_ang = sess.run([ops['merged'], ops['step'],
                                                      ops['loss'], ops['experts_prob'],
                                                      ops['normal_pred'], ops['cos_ang']], feed_dict=feed_dict)


        expert_to_use = np.argmax(experts_prob, axis=0)
        cos = cos_ang[expert_to_use, range(len(expert_to_use))]

        ang_err_batch = np.rad2deg(np.arccos(np.abs(cos)))  # unoriented
        ang_err.append(ang_err_batch)

        loss_sum += loss_val
        test_writer.add_summary(summary, step)
        total_seen += BATCH_SIZE

    mean_loss = loss_sum / float(test_num_batchs)
    log_string('eval mean loss: %f' % (mean_loss))

    ang_err = np.reshape(ang_err, [n_shapes, PATCHES_PER_SHAPE])
    rms = np.sqrt(np.mean(np.square(ang_err), axis=1))
    mean_rms = np.mean(rms)
    log_string('eval mean rms: %f' % (mean_rms))
    tf.summary.scalar('Eval RMS angle', mean_rms)

    return mean_loss


if __name__ == "__main__":

    gmm = utils.get_3d_grid_gmm(subdivisions=[N_GAUSSIANS, N_GAUSSIANS, N_GAUSSIANS], variance=GMM_VARIANCE)
    pickle.dump(gmm, open(os.path.join(LOG_DIR, 'gmm.p'), "wb"))
    train(gmm)

    LOG_FOUT.close()


