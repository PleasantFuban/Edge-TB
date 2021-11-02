import io
import time
from typing import Optional, Union

import numpy as np
import tensorflow as tf
from keras.models import Model

import worker_utils

write = io.BytesIO()
cur_index = 0


def load_data(path, start_index, _len, input_shape):
    x_list = []
    y_list = []
    for i in range(_len):
        x_list.append(np.load(f'{path}/images_{start_index + i}.npy').reshape(input_shape))
        y_list.append(np.load(f'{path}/labels_{start_index + i}.npy'))
    images = np.concatenate(tuple(x_list))
    labels = np.concatenate(tuple(y_list))
    return images, labels


def train_all(model, images, labels, epochs, batch_size):
    h = model.fit(images, labels, epochs=epochs, batch_size=batch_size)
    return h.history['loss']


def train(model, images, labels, epochs, batch_size, train_len):
    global cur_index
    cur_images = images[cur_index * 500: (cur_index + 1) * 500]
    cur_labels = labels[cur_index * 500: (cur_index + 1) * 500]
    cur_index += 1
    if cur_index == train_len:
        cur_index = 0
    h = model.fit(cur_images, cur_labels, epochs=epochs, batch_size=batch_size)
    return h.history['loss']


def test(model, images, labels):
    loss, acc = model.test_on_batch(images, labels)
    return loss, acc


def test_on_batch(model, images, labels, batch_size):
    sample_number = images.shape[0]
    batch_number = sample_number // batch_size
    last = sample_number % batch_size
    total_loss, total_acc = 0.0, 0.0
    for i in range(batch_number):
        loss, acc = model.test_on_batch(images[i * batch_size:(i + 1) * batch_size],
                                        labels[i * batch_size:(i + 1) * batch_size])
        total_loss += loss * batch_size
        total_acc += acc * batch_size
    loss, acc = model.test_on_batch(images[batch_number * batch_size:],
                                    labels[batch_number * batch_size:])
    total_loss += loss * last
    total_acc += acc * last
    return total_loss / sample_number, total_acc / sample_number


def reduce_mean_loss(y_truth, y_predicted):
    return tf.reduce_mean(tf.pow(y_truth - y_predicted, 2))


def compute_gradients(model: Model, images, labels, loss_fn=reduce_mean_loss):
    with tf.GradientTape() as tape:
        logits = model(images)
        loss_value = loss_fn(labels, logits)
    gradients = tape.gradient(loss_value, model.trainable_weights)
    return gradients


def apply_gradients(target: Union[Model, list, np.array], gradients, optimizer=None, clip=None, **kwargs):
    if isinstance(target, Model):
        optimizer = target.optimizer
        real_target = target.trainable_weights
    else:
        real_target = target
    if clip:
        gradients, *_ = clip(gradients, **kwargs)
    optimizer.apply_gradients(zip(gradients, real_target))


def parse_arrays(arrays):
    w = np.load(arrays, allow_pickle=True)
    return w


# only store the weights at received_weights [0]
# and accumulate as soon as new weights are received to save space :-)
def store_weights(received_weights, new_weights, received_count):
    if received_count == 1:
        received_weights.append(new_weights)
    else:
        received_weights[0] = np.add(received_weights[0], new_weights)


def avg_weights(received_weights, received_count):
    return received_weights[0] / received_count


def assign_weights(model, weights):
    model.set_weights(weights)


def send_arrays(arrays, path: Optional[str], node_list, connect, forward=None, forward_path=None, **kwargs):
    """
    Send numpy arrays (e.g. weights, gradients) to other nodes.

    This function will determine whether to perform a direct transfer or a forward (when the node is not in {connect}).

    If you only need to send arrays to one node, use ``send_arrays_single()``.

    :param arrays: the arrays to send, which may be a numpy array or list of numpy arrays.
    :param path: the destination path.
    :param node_list: the name list of nodes to receive the arrays. If one of the nodes are named 'self', it won't send
        the arrays to the same node.
    :param connect: a dict of sender node connection (node name -> ip+port).
    :param forward: (optional) sender's forward tables for nodes that have no direct connection to.
    :param forward_path: (optional) the path to receive the arrays if it is a forward. If not specified, the forward
        will be sent to {path}.
    :param kwargs: extra information that will be sent together with the arrays, which can be later fetched in
        `request.form` in Flask. Note that 'node' (receiver node name) and 'path' (destination path) will also be
        included in the form, so if you are doing multiple forwards, you can leave the args {path} None and let
        the function find it in kwargs.
    :return: a bool indicating whether one of the nodes are named 'self' - attempting to send arrays to self.
    """
    send_self = False
    np.save(write, arrays)
    write.seek(0)
    for node in node_list:
        send_self = send_self or _send_arrays_single(write, path, node, connect, forward, forward_path, **kwargs)
        write.seek(0)
    write.truncate()
    return send_self


def send_arrays_single(arrays, path: Optional[str], node_name, connect, forward=None, forward_path=None, **kwargs):
    """
    Send numpy arrays (e.g. weights, gradients) to another node.

    This function will determine whether to perform a direct transfer or a forward (when the node is not in {connect}).

    If you need to send arrays to multiple nodes, use ``send_arrays()``.

    :param arrays: the arrays to send, which may be a numpy array or list of numpy arrays. It can also be an instance of
        `io.IOBase` containing the arrays.
    :type arrays: list, numpy array or io.IOBase
    :param path: the destination path.
    :param node_name: the name of the nodes to receive the arrays. If the node is named 'self', the function will have
        no effect.
    :param connect: a dict of sender node connection (node name -> ip+port).
    :param forward: (optional) sender's forward tables for nodes that have no direct connection to.
    :param forward_path: (optional) the path to receive the arrays if it is a forward. If not specified, the forward
        will be sent to {path}.
    :param kwargs: extra information that will be sent together with the arrays, which can be later fetched in
        `request.form` in Flask. Note that 'node' (receiver node name) and 'path' (destination path) will also be
        included in the form, so if you are doing multiple forwards, you can leave the args {path} None and let
        the function find it in kwargs.
    :return: a bool indicating whether the node is named 'self' - attempting to send arrays to self.
    """
    if node_name == 'self':
        return False
    arrays_io: io.IOBase
    if not isinstance(arrays, io.IOBase):
        np.save(write, arrays)
        write.seek(0)
        arrays_io = write
    else:
        arrays_io = arrays
    res = _send_arrays_single(arrays_io, path, node_name, connect, forward, forward_path, **kwargs)
    if not isinstance(arrays, io.IOBase):
        arrays_io.seek(0)       # write.seek(0)
        arrays_io.truncate()    # write.truncate()
    return res


def _send_arrays_single(arrays_write, path: str, node_name, connect, forward=None, forward_path=None, **kwargs) -> bool:
    node_name = node_name or kwargs.get('name')
    if node_name == 'self':
        return False
    if node_name in connect:
        address = connect[node_name]
        data = kwargs
        _send_arrays_helper(arrays_write, data, address, path)
    elif forward and (node_name in forward):
        address = forward[node_name]
        data = {'node': node_name, 'path': path}
        data.update(kwargs)     # Note that if 'node' or 'path' in kwargs, they will cover the positional args
        forward_path = forward_path or data['path']     # if forward_path not specified, send to path
        worker_utils.log(f'need {address} to forward to {node_name} {path}')
        _send_arrays_helper(arrays_write, data, address, forward_path)
    else:
        raise Exception('The node has not connected to ' + node_name)
    return True


def _send_arrays_helper(weights, data, address, path: str):
    s = time.time()
    worker_utils.send_data('POST', path, address, data=data, files={'weights': weights})
    e = time.time()
    worker_utils.log(f'send weights to {address}, cost={e - s}')


def random_selection(node_list, number):
    return np.random.choice(node_list, number, replace=False)


def log_loss(loss, _round):
    """ Log a loss message and print it for display.

    we left a comma at the end for easy positioning and extending.
    this message can be parsed by controller/ctl_utils.py, parse_log ().
    """
    message = 'Train: loss={}, round={},'.format(loss, _round)
    worker_utils.log(message)
    return message


def log_acc(acc, _round, layer=-1):
    """ Log an accuracy message and print it for display.

    we left a comma at the end for easy positioning and extending.
    this message can be parsed by controller/ctl_utils.py, parse_log ().
    """
    if layer != -1:
        message = 'Aggregate: accuracy={}, round={}, layer={},'.format(acc, _round, layer)
    else:
        message = 'Aggregate: accuracy={}, round={},'.format(acc, _round)
    worker_utils.log(message)
    return message
