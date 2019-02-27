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
    :return: placeholders
    """

    # Placeholders for the data
    n_gaussians = gmm.means_.shape[0]
    D = gmm.means_.shape[1]
    n_rads = len(radius)

    points_pl = tf.placeholder(tf.float32, shape=(batch_size, n_points * n_rads, D))
    normal_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))

    # GMM variables
    w_pl = tf.placeholder(tf.float32, shape=(n_gaussians))
    mu_pl = tf.placeholder(tf.float32, shape=(n_gaussians, D))
    sigma_pl = tf.placeholder(tf.float32, shape=(n_gaussians, D))  # diagonal

    n_effective_points = tf.placeholder(tf.uint16, shape=(batch_size, n_rads))

    return points_pl, normal_pl, w_pl, mu_pl, sigma_pl, n_effective_points


def get_model(points, w, mu, sigma, is_training, radius, bn_decay=None, weight_decay=0.005, original_n_points=None, n_experts=2, expert_dict=None):
    """
    Normal estimation architecture for Nesti-Net
    :param points: a batch of point clouds with xyz coordinates [b x n x 3]
    :param w: GMM weights
    :param mu: GMM means
    :param sigma: GMM std
    :param is_training: true / false indicating training or testing
    :param radius: list of floats indicating radius as percentage of bounding box
    :param bn_decay:
    :param weight_decay:
    :param original_n_points: The original number of points in the vicinity of the query point ( used for compensating in the 3dmfv represenation)
    :param expert_dict: dictionary with length equal to the number of experts. stores the scales to expert assignment
    :return:
            experts_prob: the probability of each expert to predict the correct normal[b x n_experts]
            net_n_est: estimated normal [b x n x 3]
            MuPS: Multi-scale point statistics representation [b x res x res x res x 20 x n_rads]
    """

    n_rads = len(radius)
    batch_size = points.get_shape()[0].value
    n_points = points.get_shape()[1].value / n_rads
    n_gaussians = w.shape[0].value
    res = int(np.round(np.power(n_gaussians, 1.0/3.0)))

    # Create multi-scale point statistics (MuPS)
    for s in range(n_rads):
        start = s * n_points
        end = start + n_points
        single_scale_3dmfv = tf_util.get_3dmfv_n_est(points[:, start:end, :], w, mu, sigma,
                                                      flatten=True, n_original_points=original_n_points[:, s])
        single_scale_3dmfv = tf.reshape(single_scale_3dmfv, [batch_size, -1, res, res, res])
        single_scale_3dmfv = tf.transpose(single_scale_3dmfv, [0, 2, 3, 4, 1])
        if s == 0:
            MuPS = single_scale_3dmfv
        else:
            MuPS = tf.concat([MuPS, single_scale_3dmfv], axis=-1)

    experts_prob = scale_manager_net(MuPS, bn_decay, is_training, weight_decay,
                              scope_str='noise', n_experts=n_experts, n_gaussians=n_gaussians)

    experts = []
    if expert_dict is None:
        # Default expert assignment divides the scales equally between experts and adds multi-scale experts
        # for the remainder of the expert to radius ratio
        expert_to_rad_ratio = n_experts // n_rads
        expert_to_rad_ratio_modulo = n_experts % n_rads
        expert_index = 0
        expert_assignment_list = []
        for i in range(n_rads):
            for j in range(expert_to_rad_ratio):
                expert_assignment_list.append([expert_index])
            expert_index = expert_index + 1
        for i in range(expert_to_rad_ratio_modulo):
            expert_assignment_list.append(range(n_rads))
        expert_dict = {i: expert_assignment_list[i] for i in range(n_experts)}
    elif n_experts != len(expert_dict):
        ValueError('Incompatible expert assignment values in variable expert_dict ')

    for i in range(n_experts):
        start = np.min(expert_dict[i]) * 20
        end = start + 20 * (len(expert_dict[i]))
        experts.append(normal_est_net(MuPS[:, :, :, :, start:end], bn_decay, is_training, weight_decay,
                                      scope_str='Expert_' + str(i), n_gaussians=n_gaussians, divider=len(expert_dict[i]))))

    n_est = tf.stack(experts)


    return experts_prob, n_est, MuPS


def get_loss(n_pred, n_gt, experts_prob, loss_type='cos', n_experts=2, expert_type='simple'):
    """
    Given a GT normal and a predicted normal - compute the loss function
    :param n_pred: predicted normal [b x 3]
    :param n_gt: ground truth normal [b x 3]
    :param loss_type: cos/sin/euclidean distance functions for loss
    :partam expert_prob: the probability of each expert to predict the correct normal[b x n_experts]
    :return:
        loss: mean loss over all batches
    """

    n_gt = tf.tile(tf.expand_dims(n_gt, axis=0), [n_experts, 1, 1])

    # Angular Expert loss
    n_pred = tf.divide(n_pred, tf.tile(tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(n_pred), axis=2)), axis=-1), [1, 1, 3]))
    n_gt = tf.divide(n_gt, tf.tile(tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(n_gt), axis=2)), axis=-1), [1, 1, 3]))

    cos_ang = tf.reduce_sum(tf.multiply(n_pred, n_gt), axis=2)  # change after enforcing normalization
    one_minus_cos = 1.0 - tf.abs(cos_ang)
    if loss_type == 'cos':
        bool_cond = tf.greater(one_minus_cos, 0.01)
        diff = tf.where(bool_cond, one_minus_cos, 100 * tf.pow(one_minus_cos, 2))
    elif loss_type == 'euclidean':
        diff = tf.minimum(tf.reduce_sum(tf.square(n_gt - n_pred), axis=2),
                          tf.reduce_sum(tf.square(n_gt + n_pred), axis=2))
    elif loss_type == 'sin':
        diff = 2 * tf.norm(tf.cross(n_pred, n_gt), axis=2)
    else:
        ValueError('Wrong loss type...')

    if expert_type == 'simple':
        loss = tf.reduce_sum(tf.multiply(experts_prob, diff), axis=0)  # sum over experts
        loss = tf.reduce_mean(loss)
        tf.summary.scalar('Mixture of experts simple loss', loss)
    elif expert_type == 'gaussian':
        loss = -tf.log(tf.reduce_sum(tf.multiply(experts_prob, (1.0/(2*np.pi)) * tf.exp(-0.5 * tf.square(diff))), axis=0))  # sum over experts
        loss = tf.reduce_mean(loss)
        tf.summary.scalar('Mixture of experts gaussian loss', loss)
    else:
        ValueError('Wrong expert loss type...')

    return loss, cos_ang


def scale_manager_net(MuPS, bn_decay, is_training, weight_decay, scope_str, n_experts=2, n_gaussians=8):
    """
    Predict the probability of each expert to estimate the correct normalbased on the MuPS representation
    """

    if n_gaussians == 8*8*8:
        global_feature = conv_net_8g(MuPS, bn_decay, is_training, 'gating_conv')
    elif n_gaussians == 3*3*3:
        global_feature = conv_net_3g(MuPS, bn_decay, is_training, 'gating_conv')
    else:
        raise ValueError('Incompatible number of Gaussians - currently 3 and 8 are supported. '
                         'For other values you should tweak the architecture') # rais error for unsupported number of Gaussians

    net = tf_util.fully_connected(global_feature, 1024, bn=True, is_training=is_training,
                                  scope='fc1' + scope_str, bn_decay=bn_decay, weigth_decay=weight_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2' + scope_str, bn_decay=bn_decay, weigth_decay=weight_decay)
    net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training,
                                  scope='fc3' + scope_str, bn_decay=bn_decay, weigth_decay=weight_decay)
    net = tf_util.fully_connected(net, n_experts, activation_fn=tf.nn.relu, scope='fc4' + scope_str, is_training=is_training,
                                  weigth_decay=weight_decay)

    net = tf.squeeze(tf.nn.softmax(net))  # output probabilities
    net = tf.transpose(net, perm=[1, 0])  # n_experts X B
    return net

def conv_net_8g(grid_3dmfv, bn_decay, is_training, scope_str):
    """
    3D expert convolutional architecture for 8 Gaussians
    """

    batch_size = grid_3dmfv.get_shape()[0].value

    layer = 1
    net = inception_module(grid_3dmfv, n_filters=128, kernel_sizes=[3, 5], is_training=is_training,
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
    net = inception_module(net, n_filters=512, kernel_sizes=[2, 4], is_training=is_training, bn_decay=bn_decay,
                           scope='inception' + str(layer) + scope_str)
    layer = layer + 1
    net = inception_module(net, n_filters=512, kernel_sizes=[2, 4], is_training=is_training, bn_decay=bn_decay,
                           scope='inception' + str(layer) + scope_str)
    layer = layer + 1
    net = tf_util.max_pool3d(net, [2, 2, 2], scope='maxpool' + str(layer) + scope_str, stride=[2, 2, 2], padding='SAME')
    layer = layer + 1
    net = inception_module(net, n_filters=512, kernel_sizes=[1, 2], is_training=is_training, bn_decay=bn_decay,
                           scope='inception' + str(layer) + scope_str)
    layer = layer + 1
    net = tf_util.max_pool3d(net, [2, 2, 2], scope='maxpool' + str(layer) + scope_str, stride=[2, 2, 2], padding='SAME')

    global_feature = tf.reshape(net, [batch_size, -1])

    return global_feature

def conv_net_3g(grid_3dmfv, bn_decay, is_training, scope_str):
    """
    3D expert convolutional architecture for 3 Gaussians
    """

    batch_size = grid_3dmfv.get_shape()[0].value

    layer = 1
    net = inception_module(grid_3dmfv, n_filters=128, kernel_sizes=[2, 3], is_training=is_training,
                           bn_decay=bn_decay, scope='inception' + str(layer) + scope_str)
    layer = layer + 1
    net = inception_module(net, n_filters=256, kernel_sizes=[2, 3], is_training=is_training, bn_decay=bn_decay,
                           scope='inception' + str(layer) + scope_str)
    layer = layer + 1
    net = inception_module(net, n_filters=256, kernel_sizes=[1, 2], is_training=is_training, bn_decay=bn_decay,
                           scope='inception' + str(layer) + scope_str)
    layer = layer + 1
    net = inception_module(net, n_filters=512, kernel_sizes=[1, 2], is_training=is_training, bn_decay=bn_decay,
                           scope='inception' + str(layer) + scope_str)
    layer = layer + 1
    net = tf_util.max_pool3d(net, [3, 3, 3], scope='maxpool' + str(layer) + scope_str, stride=[2, 2, 2], padding='SAME')

    global_feature = tf.reshape(net, [batch_size, -1])
    return global_feature


def normal_est_net(grid_fisher, bn_decay, is_training, weight_decay, scope_str, n_gaussians, divider=1 ):
    """
    Expert architecture
    """

    # 3D CNN architecture - adjust architecture to number of Gaussians
    if n_gaussians == 8*8*8:
        # a lighter version of conv_net_8g
        batch_size = grid_fisher.get_shape()[0].value
        layer = 1
        # divider is used to balance the size of experts (maintain the same capacity)
        net = inception_module(grid_fisher, n_filters=np.round(128 / divider), kernel_sizes=[3, 5], is_training=is_training,
                               bn_decay=bn_decay, scope='inception' + str(layer) + scope_str)
        layer = layer + 1
        net = inception_module(net, n_filters=256, kernel_sizes=[3, 5], is_training=is_training, bn_decay=bn_decay,
                               scope='inception' + str(layer) + scope_str)
        layer = layer + 1
        net = tf_util.max_pool3d(net, [2, 2, 2], scope='maxpool' + str(layer) + scope_str, stride=[2, 2, 2], padding='SAME')

        layer = layer + 1
        net = inception_module(net, n_filters=256, kernel_sizes=[2, 4], is_training=is_training, bn_decay=bn_decay,
                               scope='inception' + str(layer) + scope_str)
        layer = layer + 1
        net = tf_util.max_pool3d(net, [2, 2, 2], scope='maxpool' + str(layer) + scope_str, stride=[2, 2, 2], padding='SAME')
        layer = layer + 1
        net = inception_module(net, n_filters=512, kernel_sizes=[2, 4], is_training=is_training, bn_decay=bn_decay,
                               scope='inception' + str(layer) + scope_str)
        layer = layer + 1
        net = tf_util.max_pool3d(net, [2, 2, 2], scope='maxpool' + str(layer) + scope_str, stride=[2, 2, 2], padding='SAME')

        global_feature = tf.reshape(net, [batch_size, -1])
    elif n_gaussians == 3*3*3:
        global_feature = conv_net_3g(grid_fisher, bn_decay, is_training, scope_str +'_expert_conv')
    else:
        raise ValueError('Incompatible number of Gaussians') # will throw error

    #  FC architectrure - normal estiamation network
    net = tf_util.fully_connected(global_feature, 512, bn=True, is_training=is_training,
                                  scope='fc1' + scope_str, bn_decay=bn_decay, weigth_decay=weight_decay)
    net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training,
                                  scope='fc2' + scope_str, bn_decay=bn_decay, weigth_decay=weight_decay)
    net = tf_util.fully_connected(net, 64, bn=True, is_training=is_training,
                                  scope='fc3' + scope_str, bn_decay=bn_decay, weigth_decay=weight_decay)
    net = tf_util.fully_connected(net, 3, activation_fn=None, scope='fc4' + scope_str, is_training=is_training,
                                  weigth_decay=weight_decay)

    net = tf.squeeze(net)

    return net


def inception_module(input, n_filters=64, kernel_sizes=[3, 5], is_training=None, bn_decay=None, scope='inception'):
    """
     3D inception_module
     """
    one_by_one =  tf_util.conv3d(input, n_filters, [1, 1, 1], scope= scope + '_conv1',
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



if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 1024, 3))
        outputs = get_model(inputs, tf.constant(True))
        print (outputs)
