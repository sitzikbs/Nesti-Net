import tensorflow as tf
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util


def placeholder_inputs(batch_size, n_points, gmm, radius,  testing=False):
    """
    initialize placeholders for the inputs
    :param batch_size:
    :param n_points: number of points in each point cloud
    :param gmm: Gausian Mixtuere Model (GMM) sklearn object
    :param radius: a list of radii
    :param testing: True / False - enabling different number of input points while testing
    :return:
    """

    #Placeholders for the data
    n_gaussians = gmm.means_.shape[0]
    D = gmm.means_.shape[1]
    n_rads = len(radius)

    if testing == True:
        points_pl = tf.placeholder(tf.float32, shape=(batch_size, None, D))
        n_est_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))
    else:
        points_pl = tf.placeholder(tf.float32, shape=(batch_size, n_points * n_rads, D))
        n_est_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))

    #GMM variables
    w_pl = tf.placeholder(tf.float32, shape=(n_gaussians))
    mu_pl = tf.placeholder(tf.float32, shape=(n_gaussians, D))
    sigma_pl = tf.placeholder(tf.float32, shape=(n_gaussians, D))  # diagonal

    n_effective_points = tf.placeholder(tf.uint16, shape=(batch_size, n_rads))

    return points_pl, n_est_pl, w_pl, mu_pl, sigma_pl, n_effective_points


def get_model(points, w, mu, sigma, is_training, radius, bn_decay=None, weight_decay=0.005, original_n_points=None):
    """
    Normal estimation architecture for multi-scale and single-scale (ms, ss) Nesti-Net Ablations
    :param points: a batch of point clouds with xyz coordinates [b x n x 3]
    :param w: GMM weights
    :param mu: GMM means
    :param sigma: GMM std
    :param is_training: true / false indicating training or testing
    :param radius: list of floats indicating radius as percentage of bounding box
    :param bn_decay:
    :param weight_decay:
    :param original_n_points: The original number of points in the vicinity of the query point ( used for compensating in the 3dmfv represenation)
    :return:
            net_n_est: estimated normal [b x n x 3]
            grid_fisher: 3dmfv representation of each points cloud in the batch
    """

    batch_size = points.get_shape()[0].value
    n_rads = len(radius)
    n_points = points.get_shape()[1].value / n_rads
    n_gaussians = w.shape[0].value
    res = int(np.round(np.power(n_gaussians, 1.0/3.0)))

    for s, rad in enumerate(radius):
        start = s * n_points
        end = start + n_points
        if original_n_points is None:
            fv = tf_util.get_3dmfv_n_est(points[:, start:end, :], w, mu, sigma, flatten=True, n_original_points=None)
        else:
            fv = tf_util.get_3dmfv_n_est(points[:, start:end, :], w, mu, sigma, flatten=True, n_original_points=original_n_points[:, s])

        if s == 0:
            grid_fisher = tf.reshape(fv, [batch_size, -1, res, res, res])
        else:
            grid_fisher = tf.concat([grid_fisher, tf.reshape(fv, [batch_size, -1, res, res, res])], axis=1)
    grid_fisher = tf.transpose(grid_fisher, [0, 2, 3, 4, 1])

    #CNN architecture - adjust architecture to number of gaussians
    if n_gaussians == 8*8*8:
        layer = 1
        net = inception_module(grid_fisher, n_filters=128, kernel_sizes=[3, 5], is_training=is_training, bn_decay=bn_decay, scope='inception_s'+str(s)+'_l_'+str(layer))
        layer = layer + 1
        net = inception_module(net, n_filters=256, kernel_sizes=[3, 5], is_training=is_training, bn_decay=bn_decay, scope='inception_s'+str(s)+'_l_'+str(layer))
        layer = layer + 1
        net = inception_module(net, n_filters=256, kernel_sizes=[3, 5], is_training=is_training, bn_decay=bn_decay, scope='inception_s'+str(s)+'_l_'+str(layer))
        layer = layer + 1
        net = tf_util.max_pool3d(net, [2, 2, 2], scope='maxpool_s'+str(s)+'_l_'+str(layer), stride=[2, 2, 2], padding='SAME')
        layer = layer + 1
        net = inception_module(net, n_filters=512, kernel_sizes=[3, 4], is_training=is_training, bn_decay=bn_decay, scope='inception_s'+str(s)+'_l_'+str(layer))
        layer = layer + 1
        net = inception_module(net, n_filters=512, kernel_sizes=[3, 4], is_training=is_training, bn_decay=bn_decay, scope='inception_s'+str(s)+'_l_'+str(layer))
        layer = layer + 1
        net = tf_util.max_pool3d(net, [2, 2, 2], scope='maxpool_s'+str(s)+'_l_'+str(layer), stride=[2, 2, 2], padding='SAME')
        global_feature = tf.reshape(net, [batch_size, -1])
    elif n_gaussians == 3 * 3 * 3:
        layer = 1
        net = inception_module(grid_fisher, n_filters=128, kernel_sizes=[2, 3], is_training=is_training,
                               bn_decay=bn_decay, scope='inception_s'+str(s)+'_l_'+str(layer))
        layer = layer + 1
        net = inception_module(net, n_filters=256, kernel_sizes=[2, 3], is_training=is_training, bn_decay=bn_decay,
                               scope='inception_s'+str(s)+'_l_'+str(layer))
        layer = layer + 1
        net = inception_module(net, n_filters=256, kernel_sizes=[1, 2], is_training=is_training, bn_decay=bn_decay,
                               scope='inception_s'+str(s)+'_l_'+str(layer))
        layer = layer + 1
        net = inception_module(net, n_filters=512, kernel_sizes=[1, 2], is_training=is_training, bn_decay=bn_decay,
                               scope='inception_s'+str(s)+'_l_'+str(layer))
        layer = layer + 1
        net = tf_util.max_pool3d(net, [3, 3, 3], scope='maxpool_s'+str(s)+'_l_'+str(layer), stride=[2, 2, 2],
                                 padding='SAME')

        global_feature = tf.reshape(net, [batch_size, -1])
    else:
      raise ValueError('Unsupported number of Gaussians - you should change the architecture accordingly')  # will throw error if using an unsupported number of gaussians

    #  FC architectrure - normal estiamation network
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

    net_n_est = tf.squeeze(net_n_est)
    if batch_size == 1:
        net_n_est = tf.expand_dims(net_n_est, axis=0)

    return net_n_est, grid_fisher

def inception_module(input, n_filters=64, kernel_sizes=[3,5], is_training=None, bn_decay=None, scope='inception'):
    """
    3D inception_module
    """
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
    average_pooling = tf_util.conv3d(average_pooling, n_filters, [1, 1, 1], scope= scope + '_conv4',
           stride=[1, 1, 1], padding='SAME', bn=True,
           bn_decay=bn_decay, is_training=is_training)

    output = tf.concat([ one_by_one, three_by_three, five_by_five, average_pooling], axis=4)
    #output = output + tf.tile(input) ??? #resnet
    return output



def get_loss(n_pred, n_gt, loss_type='cos'):
    """
    Given a GT normal and a predicted normal - compute the loss function
    :param n_pred: predicted normal [b x 3]
    :param n_gt: ground truth normal [b x 3]
    :param loss_type: cos/sin/euclidean distance functions for loss
    :return:
        loss: mean loss over all batches
    """

    n_pred = tf.divide(n_pred, tf.tile(tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(n_pred), axis=1)), axis=-1),
                                       [1, 3]))
    n_gt = tf.divide(n_gt,
                       tf.tile(tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(n_gt), axis=1)), axis=-1), [1, 3]))
    cos_ang = tf.reduce_sum(tf.multiply(n_pred, n_gt), axis=1)  # change after enforcing normalization
    one_minus_cos = 1.0 - tf.abs(cos_ang)
    if loss_type == 'cos':
        bool_cond = tf.greater(one_minus_cos, 0.01)
        all_losses = tf.where(bool_cond, one_minus_cos, 100 * tf.pow(one_minus_cos, 2))
        loss = tf.reduce_mean(all_losses)
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
        ValueError('Unsupported loss type...')

    return loss, cos_ang

if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 1024, 3))
        outputs = get_model(inputs, tf.constant(True))
        print (outputs)
