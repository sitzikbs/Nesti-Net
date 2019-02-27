import tensorflow as tf
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util


def placeholder_inputs(batch_size, n_points, gmm, testing=False):

    #Placeholders for the data
    n_gaussians = gmm.means_.shape[0]
    D = gmm.means_.shape[1]

    if testing == True:
        points_pl = tf.placeholder(tf.float32, shape=(batch_size, None, D))
        n_est_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))
    else:
        points_pl = tf.placeholder(tf.float32, shape=(batch_size, n_points, D))
        n_est_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))

    #GMM variables
    w_pl = tf.placeholder(tf.float32, shape=(n_gaussians))
    mu_pl = tf.placeholder(tf.float32, shape=(n_gaussians, D))
    sigma_pl = tf.placeholder(tf.float32, shape=(n_gaussians, D)) # diagonal

    n_effective_points = tf.placeholder(tf.uint16, shape=(batch_size,))

    return points_pl, n_est_pl, w_pl, mu_pl, sigma_pl, n_effective_points


def get_model(points, w, mu, sigma, is_training, bn_decay=None, weight_decay=0.005, original_n_points=None, labels=None):
    """ Classification PFV-Network, input is BxNx3"""

    batch_size = points.get_shape()[0].value
    n_points = points.get_shape()[1].value
    n_gaussians = w.shape[0].value
    res = int(np.round(np.power(n_gaussians, 1.0/3.0)))


    if original_n_points is None:
        fv = tf_util.get_3dmfv_n_est(points, w, mu, sigma, flatten=True, n_original_points=None)
    else:
        fv = tf_util.get_3dmfv_n_est(points, w, mu, sigma, flatten=True, n_original_points=original_n_points)

    grid_fisher = tf.reshape(fv, [batch_size, -1, res, res, res])
    grid_fisher = tf.transpose(grid_fisher, [0, 2, 3, 4, 1])

    # Inception
    layer = 1
    net = inception_module(grid_fisher, n_filters=128, kernel_sizes=[3, 5], is_training=is_training, bn_decay=bn_decay, scope='inception'+str(layer))
    layer = layer + 1
    net = inception_module(net, n_filters=256, kernel_sizes=[3, 5], is_training=is_training, bn_decay=bn_decay, scope='inception'+str(layer))
    layer = layer + 1
    net = inception_module(net, n_filters=256, kernel_sizes=[3, 5], is_training=is_training, bn_decay=bn_decay, scope='inception'+str(layer))
    layer = layer + 1
    net = tf_util.max_pool3d(net, [2, 2, 2], scope='maxpool'+str(layer), stride=[2, 2, 2], padding='SAME')
    layer = layer + 1
    net = inception_module(net, n_filters=512, kernel_sizes=[3, 5], is_training=is_training, bn_decay=bn_decay, scope='inception'+str(layer))
    layer = layer + 1
    net = inception_module(net, n_filters=512, kernel_sizes=[3, 5], is_training=is_training, bn_decay=bn_decay, scope='inception'+str(layer))
    layer = layer + 1
    net = tf_util.max_pool3d(net, [2, 2, 2], scope='maxpool'+str(layer), stride=[2, 2, 2], padding='SAME')

    global_feature = tf.reshape(net, [batch_size, -1])

    #normal estiamation  network


    net = tf_util.fully_connected(global_feature, 1024, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay, weigth_decay=weight_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay, weigth_decay=weight_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training,
                                  scope='fc3', bn_decay=bn_decay, weigth_decay=weight_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp3')
    net_n_est = tf_util.fully_connected(net, 3, activation_fn=None, scope='fc4', is_training=is_training,
                                  weigth_decay=weight_decay)

    # TO DO - Add constrint to enforce unit size
    net_n_est = tf.squeeze(net_n_est)
    if batch_size == 1:
        net_n_est = tf.expand_dims(net_n_est, axis=0)
    return net_n_est, grid_fisher

def inception_module(input, n_filters=64, kernel_sizes=[3,5], is_training=None, bn_decay=None, scope='inception'):
    one_by_one =  tf_util.conv3d(input, n_filters, [1,1,1], scope= scope + '_conv1',
           stride=[1, 1, 1], padding='SAME', bn=True,
           bn_decay=bn_decay, is_training=is_training)
    three_by_three = tf_util.conv3d(one_by_one, int(n_filters/2), [kernel_sizes[0], kernel_sizes[0], kernel_sizes[0]], scope= scope + '_conv2',
           stride=[1, 1, 1], padding='SAME', bn=True,
           bn_decay=bn_decay, is_training=is_training)
    five_by_five = tf_util.conv3d(one_by_one, int(n_filters/2), [kernel_sizes[1], kernel_sizes[1], kernel_sizes[1]], scope=scope + '_conv3',
                          stride=[1, 1, 1], padding='SAME', bn=True,
                          bn_decay=bn_decay, is_training=is_training)
    average_pooling = tf_util.avg_pool3d(input, [kernel_sizes[0], kernel_sizes[0], kernel_sizes[0]], scope=scope+'_avg_pool', stride=[1, 1, 1], padding='SAME')
    average_pooling = tf_util.conv3d(average_pooling, n_filters, [1,1,1], scope= scope + '_conv4',
           stride=[1, 1, 1], padding='SAME', bn=True,
           bn_decay=bn_decay, is_training=is_training)

    output = tf.concat([ one_by_one, three_by_three, five_by_five, average_pooling], axis=4)
    #output = output + tf.tile(input) ??? #resnet
    return output



def get_loss(n_pred, n_gt, loss_type='cos',):

    # normal estimation loss
    # n_pred = tf.nn.l2_normalize(n_pred, dim=1)
    n_pred = tf.divide(n_pred, tf.tile(tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(n_pred), axis=1)), axis=-1), [1,3]))
    n_gt = tf.nn.l2_normalize(n_gt, dim=1)
    cos_ang = tf.reduce_sum(tf.multiply(n_pred, n_gt), axis=1)  # change after enforcing normalization
    one_minus_cos = 1.0 - tf.abs(cos_ang)
    if loss_type == 'cos':
        bool_cond = tf.greater(one_minus_cos, 0.01)
        all_losses = tf.where(bool_cond, one_minus_cos, 100 * tf.pow(one_minus_cos, 2))
        loss = tf.reduce_mean(all_losses)

        # loss = tf.reduce_mean(tf.square(1 - tf.abs(cos_ang))) # unoriented normals loss (abs)
        tf.summary.scalar('normal_estimation_loss - cos', loss)
    elif loss_type == 'euclidean':
        loss = tf.minimum(tf.reduce_sum(tf.square(n_gt - n_pred), axis=1),
                          tf.reduce_sum(tf.square(n_gt + n_pred), axis=1))
        loss = tf.reduce_mean(loss)
        tf.summary.scalar('normal_estimation_loss - euclidean', loss)
    elif loss_type == 'sin':
        sin = 2 * tf.norm(tf.cross(n_pred, n_gt), axis=1)
        loss = tf.reduce_mean(sin)
        tf.summary.scalar('normal_estimation_loss - sin', loss)
    else:
        ValueError('Wrong loss type...')

    return loss, cos_ang

if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 1024, 3))
        outputs = get_model(inputs, tf.constant(True))
        print (outputs)
