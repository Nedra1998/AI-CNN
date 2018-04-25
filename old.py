from datetime import datetime
import time

import numpy as np
import tensorflow as tf

import data



def cnn_model_fn(features, labels, mode):
    input_layer = features['x']
    logits = tf.layers.dense(inputs=input_layer, units=20)
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy":
        tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def get_label(labels):
    new = np.zeros(labels.shape[1])
    for i, row in enumerate(labels.T):
        new[i] = np.where(row==1)[0]
    return new

def main(argv=None):
    train_data, train_labels, _ = data.load_data([1, 2, 3, 4])
    eval_data, eval_labels, _ = data.load_data([5])
    train_labels = get_label(train_labels)
    eval_labels = get_label(eval_labels)

    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/cnn")

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    train_input = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data.T.astype('float32')},
        y=train_labels.astype('int32'),
        batch_size=100,
        num_epochs=50,
        shuffle=True)
    classifier.train(
        input_fn=train_input, steps=20000, hooks=[logging_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data.T.astype('float32')}, y=eval_labels.astype('int32'), num_epochs=1, shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == '__main__':
    tf.app.run()
