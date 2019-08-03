# coding: utf8
# implement a Echo-RNN network. It just echo the input which fall behind 3 steps.
# https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

batch_size = 5
num_epochs = 10000

total_series_length = 500000
# sequance data value range
num_classes = 2

# unrolled size.
truncated_backprop_length = 25
# ?
state_size = 4
echo_step = 5

num_batches = total_series_length//batch_size//truncated_backprop_length

# Generate features and labels, label value can be 0 or 1.
# x/y.shape = [batch_size, -1] => [5, 10000], what does 10000 represent for, series size.
def generate_data():
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    # why shift 3 steps?
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))

    return (x, y)

#
# placeholders for network input, series and init_state, shape->[5,15]
x = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length], name='PH_X')
y = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length], name='PH_Y')
# why set state_size to 4?
init_state = tf.placeholder(tf.float32, [batch_size, state_size], name='PH_INITSTATE')

# Unpack columns into list, each one represent an time-step as a batch. [15, 5]
# 步长为15
inputs_series = tf.unstack(x, axis=1)
labels_series = tf.unstack(y, axis=1)

# unrolling RNN, Forward pass, 15 RNN cells
current_state = init_state # [5, 4]
states_series = []
logits_series = []
predictions_series = []
for current_input in inputs_series:
    current_input = tf.reshape(current_input, [batch_size, 1])

    # Every cell deserves independent variables
    #
    # weights for current input(1) and state(4)
    W = tf.Variable(np.random.rand(state_size + 1, state_size), dtype=tf.float32, name='Weights_input_state')
    b = tf.Variable(np.zeros((1, state_size)), dtype=tf.float32, name='Bias_input_state')

    fc_weights = tf.Variable(np.random.rand(state_size, num_classes), dtype=tf.float32, name='Weights_fc')
    fc_bias = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float32, name='Bias_fc')

    # Increasing number of columns, [batch_size, state_size + 1], [5, 5]
    input_and_state_concatenated = tf.concat([current_input, current_state], axis=1)
    # rnn core part: combine previous state/output and current input.[5, 5] * [5, 4] => 5, 4
    # 所有time step共享参数？
    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition
    # state to logits [fc layer] for classification. 5,4 * 4,2 => 5,2
    logits = tf.matmul(current_state, fc_weights) + fc_bias
    predictions = tf.nn.softmax(logits)

    states_series.append(next_state)
    logits_series.append(logits)
    predictions_series.append(predictions)

    current_state = next_state

class_series = [tf.argmax(logits, axis=1) for logits in logits_series]
acc_op_series = [tf.metrics.accuracy(labels=lb, predictions=pred) for lb, pred in zip(labels_series, class_series) ]
acc_series = [acc_op[0] for acc_op in acc_op_series]
acc_ops = [acc_op[1] for acc_op in acc_op_series]
total_acc = tf.reduce_mean(acc_series)

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    print(shape)
    print(len(shape))
    variable_parameters = 1
    for dim in shape:
        print(dim)
        variable_parameters *= dim.value
    print(variable_parameters)
    total_parameters += variable_parameters
print(total_parameters)


def plot(loss_list, acc_list, predictions_series, batchX, batchY):
    plt.subplot(2, 4, 1)
    plt.cla()
    plt.plot(loss_list)
    plt.subplot(2, 4, 2)
    plt.cla()
    plt.plot(acc_list)

    for batch_series_idx in range(batch_size):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

        plt.subplot(2, 4, batch_series_idx + 3)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)


with tf.Session() as sess:
    writer = tf.summary.FileWriter('./log', sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []
    acc_list = []

    for epoch_idx in range(num_epochs):
        _x,_y = generate_data()
        _current_state = np.zeros((batch_size, state_size))

        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = _x[:,start_idx:end_idx]
            batchY = _y[:,start_idx:end_idx]

            if not batch_idx % 1000:
                _total_loss, _, _total_acc, _train_step, _current_state, _predictions_series = sess.run(
                    [total_loss, acc_ops, total_acc, train_step, current_state, predictions_series],
                    feed_dict={
                        x:batchX,
                        y:batchY,
                        init_state:_current_state
                    })

                loss_list.append(_total_loss)
                acc_list.append(_total_acc)

                print("Step",batch_idx, "Loss", _total_loss, "Acc", _total_acc)
                plot(loss_list, acc_list, _predictions_series, batchX, batchY)
            else:
                _train_step, _current_state, _predictions_series = sess.run(
                [train_step, current_state, predictions_series],
                feed_dict={
                    x: batchX,
                    y: batchY,
                    init_state: _current_state
                })

plt.ioff()
