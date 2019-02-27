import tensorflow as tf
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util


def placeholder_inputs(batch_size, n_points, gmm, radius):
    """
    initialize placeholders for the inputs
    :param batch_size:
    :param n_points: number of points in each point cloud
    :param gmm: Gausian Mixtuere Model (GMM) sklearn object
    :param radius: a list of radii
    :return:
    """

    #Placeholders for the data
    n_gaussians = gmm.means_.shape[0]
    D = gmm.means_.shape[1]
    n_rads = len(radius)

    points_pl = tf.placeholder(tf.float32, shape=(batch_size, n_points * n_rads, D))
    noise_est_pl = tf.placeholder(tf.float32, shape=(batch_size,))
    normal_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))

    #GMM variables
    w_pl = tf.placeholder(tf.float32, shape=(n_gaussians))
    mu_pl = tf.placeholder(tf.float32, shape=(n_gaussians, D))
    sigma_pl = tf.placeholder(tf.float32, shape=(n_gaussians, D))  # diagonal

    n_effective_points = tf.placeholder(tf.uint16, shape=(batch_size, n_rads))

    return points_pl, noise_est_pl, normal_pl, w_pl, mu_pl, sigma_pl, n_effective_points


def get_model(points, w, mu, sigma, is_training, radius, bn_decay=None, weight_decay=0.005, original_n_points=None):

    """
    Normal estimation architecture for learned switching by internally evaluating noise level
    :param points: a batch of point clouds with xyz coordinates [b x n x 3]
    :param w: GMM weights
    :param mu: GMM means
    :param sigma: GMM std
    :param is_training: true / false indicating training or testing
    :param radius: list of floats indicating radius as percentage of bounding box (currently only supports 2 radii)
    :param bn_decay:
    :param weight_decay:
    :param original_n_points: The original number of points in the vicinity of the query point ( used for compensating
     in the 3dmfv represenation)
    :return:
            noise_est: estimated noise level [b]
            net_n_est: estimated normal [b x n x 3]
            grid_fisher_large_scale: 3dmfv representation of each large scale points cloud in the batch
    """

    n_rads = len(radius)
    batch_size = points.get_shape()[0].value
    n_points = points.get_shape()[1].value / n_rads
    n_gaussians = w.shape[0].value
    res = int(np.round(np.power(n_gaussians, 1.0/3.0)))


    fv_large_scale = tf_util.get_3dmfv_n_est(points[:, n_points:n_points*2, :], w, mu, sigma, flatten=True, n_original_points=original_n_points[:,1])
    grid_fisher_large_scale = tf.reshape(fv_large_scale, [batch_size, -1, res, res, res])
    grid_fisher_large_scale = tf.transpose(grid_fisher_large_scale, [0, 2, 3, 4, 1])

    fv_small_scale = tf_util.get_3dmfv_n_est(points[:, 0:n_points, :], w, mu, sigma, flatten=True, n_original_points=original_n_points[:,0])
    grid_fisher_small_scale = tf.reshape(fv_small_scale, [batch_size, -1, res, res, res])
    grid_fisher_small_scale = tf.transpose(grid_fisher_small_scale, [0, 2, 3, 4, 1])


    noise_est = noise_est_net(grid_fisher_large_scale, bn_decay, is_training, weight_decay, scope_str='noise')

    n_est_large = normal_est_net(grid_fisher_large_scale, bn_decay, is_training, weight_decay, scope_str='large')
    n_est_small = normal_est_net(grid_fisher_small_scale, bn_decay, is_training, weight_decay, scope_str='small')

    mask = noise_est < 0.015

    n_est = tf.where(mask, n_est_small, n_est_large)

    if batch_size == 1:
        noise_est = tf.expand_dims(noise_est, axis=0)

    return noise_est, n_est, grid_fisher_large_scale



def get_loss(noise_pred, noise_gt, n_pred, n_gt, loss_type='cos'):
    """
    Given a GT normal and a predicted normal - compute the loss function
    :param noise_pred: predicted noise level [b]
    :param noise_gt:ground truth noise level [b]
    :param n_pred: predicted normal [b x 3]
    :param n_gt: ground truth normal [b x 3]
    :param loss_type: cos/sin/euclidean distance functions for loss
    :return:
        loss: mean loss over all batches
        cos_ang: cosine of the angle between n_pred and n_gt
    """

    # Noise loss
    noise_loss = tf.reduce_mean(tf.pow(noise_pred - noise_gt, 2), axis=0)
    tf.summary.scalar('Noise estimation loss', noise_loss)

    # Angular loss
    n_pred = tf.divide(n_pred, tf.tile(tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(n_pred), axis=1)), axis=-1), [1, 3]))
    n_gt = tf.nn.l2_normalize(n_gt, axis=1)
    cos_ang = tf.reduce_sum(tf.multiply(n_pred, n_gt), axis=1)  # change after enforcing normalization
    one_minus_cos = 1.0 - tf.abs(cos_ang)
    if loss_type == 'cos':
        bool_cond = tf.greater(one_minus_cos, 0.01)
        all_losses = tf.where(bool_cond, one_minus_cos, 100 * tf.pow(one_minus_cos, 2))
        angle_loss = tf.reduce_mean(all_losses)

        # loss = tf.reduce_mean(tf.square(1 - tf.abs(cos_ang))) # unoriented normals loss (abs)
        tf.summary.scalar('normal_estimation_loss - cos', angle_loss)
    elif loss_type == 'euclidean':
        angle_loss = tf.minimum(tf.reduce_sum(tf.square(n_gt - n_pred), axis=1),
                          tf.reduce_sum(tf.square(n_gt + n_pred), axis=1))
        angle_loss = tf.reduce_mean(angle_loss)
        tf.summary.scalar('normal_estimation_loss - euclidean', angle_loss)
    elif loss_type == 'sin':
        sin = 2 * tf.norm(tf.cross(n_pred, n_gt), axis=1)
        angle_loss = tf.reduce_mean(sin)
        tf.summary.scalar('normal_estimation_loss - sin', angle_loss)
    else:
        ValueError('Wrong loss type...')

    loss = noise_loss + angle_loss
    return loss, cos_ang


def noise_est_net(grid_fisher, bn_decay, is_training, weight_decay, scope_str):
    batch_size = grid_fisher.get_shape()[0].value
    layer = 1
    net = inception_module(grid_fisher, n_filters=128, kernel_sizes=[3, 5], is_training=is_training,
                           bn_decay=bn_decay, scope='inception' + str(layer) + scope_str)
    layer = layer + 1
    net = inception_module(net, n_filters=256, kernel_sizes=[3, 5], is_training=is_training, bn_decay=bn_decay,
                           scope='inception' + str(layer) + scope_str)
    layer = layer + 1
    net = inception_module(net, n_filters=256, kernel_sizes=[3, 5], is_training=is_training, bn_decay=bn_decay,
                           scope='inception' + str(layer) + scope_str)
    layer = layer + 1
    net = tf_util.max_pool3d(net, [2, 2, 2], scope='maxpool' + str(layer) + scope_str, stride=[2, 2, 2], padding='SAME')
    layer = layer + 1
    net = inception_module(net, n_filters=512, kernel_sizes=[3, 5], is_training=is_training, bn_decay=bn_decay,
                           scope='inception' + str(layer) + scope_str)
    layer = layer + 1
    net = inception_module(net, n_filters=512, kernel_sizes=[3, 5], is_training=is_training, bn_decay=bn_decay,
                           scope='inception' + str(layer) + scope_str)
    layer = layer + 1
    net = tf_util.max_pool3d(net, [2, 2, 2], scope='maxpool' + str(layer) + scope_str, stride=[2, 2, 2], padding='SAME')

    global_feature = tf.reshape(net, [batch_size, -1])

    # normal estiamation  network
    net = tf_util.fully_connected(global_feature, 1024, bn=True, is_training=is_training,
                                  scope='fc1' + scope_str, bn_decay=bn_decay, weigth_decay=weight_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2' + scope_str, bn_decay=bn_decay, weigth_decay=weight_decay)
    net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training,
                                  scope='fc3' + scope_str, bn_decay=bn_decay, weigth_decay=weight_decay)
    net = tf_util.fully_connected(net, 1, activation_fn=tf.nn.relu, scope='fc4' + scope_str, is_training=is_training,
                                  weigth_decay=weight_decay)
    net = tf.squeeze(net)

    return net


def normal_est_net(grid_fisher, bn_decay, is_training, weight_decay, scope_str):

    #CNN architecture - adjust architecture to number of gaussians (currently supports only 8 x 8 x 8)
    batch_size = grid_fisher.get_shape()[0].value
    layer = 1
    net = inception_module(grid_fisher, n_filters=128, kernel_sizes=[3, 5], is_training=is_training,
                           bn_decay=bn_decay, scope='inception' + str(layer) + scope_str)
    layer = layer + 1
    net = inception_module(net, n_filters=256, kernel_sizes=[3, 5], is_training=is_training, bn_decay=bn_decay,
                           scope='inception' + str(layer) + scope_str)
    layer = layer + 1
    net = inception_module(net, n_filters=256, kernel_sizes=[3, 5], is_training=is_training, bn_decay=bn_decay,
                           scope='inception' + str(layer) + scope_str)
    layer = layer + 1
    net = tf_util.max_pool3d(net, [2, 2, 2], scope='maxpool' + str(layer) + scope_str, stride=[2, 2, 2], padding='SAME')
    layer = layer + 1
    net = inception_module(net, n_filters=512, kernel_sizes=[3, 5], is_training=is_training, bn_decay=bn_decay,
                           scope='inception' + str(layer) + scope_str)
    layer = layer + 1
    net = inception_module(net, n_filters=512, kernel_sizes=[3, 5], is_training=is_training, bn_decay=bn_decay,
                           scope='inception' + str(layer) + scope_str)
    layer = layer + 1
    net = tf_util.max_pool3d(net, [2, 2, 2], scope='maxpool' + str(layer) + scope_str, stride=[2, 2, 2], padding='SAME')

    global_feature = tf.reshape(net, [batch_size, -1])

    #  FC architectrure - normal estiamation network
    net = tf_util.fully_connected(global_feature, 1024, bn=True, is_training=is_training,
                                  scope='fc1' + scope_str, bn_decay=bn_decay, weigth_decay=weight_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2' + scope_str, bn_decay=bn_decay, weigth_decay=weight_decay)
    net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training,
                                  scope='fc3' + scope_str, bn_decay=bn_decay, weigth_decay=weight_decay)
    net = tf_util.fully_connected(net, 3, activation_fn=None, scope='fc4' + scope_str, is_training=is_training,
                                  weigth_decay=weight_decay)
    net = tf.squeeze(net)

    return net


def inception_module(input, n_filters=64, kernel_sizes=[3, 5], is_training=None, bn_decay=None, scope='inception'):
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
    average_pooling = tf_util.conv3d(average_pooling, n_filters, [1,1,1], scope= scope + '_conv4',
           stride=[1, 1, 1], padding='SAME', bn=True,
           bn_decay=bn_decay, is_training=is_training)

    output = tf.concat([one_by_one, three_by_three, five_by_five, average_pooling], axis=4)
    # output = output + tf.tile(input) ??? #resnet
    return output



if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 1024, 3))
        outputs = get_model(inputs, tf.constant(True))
        print (outputs)
