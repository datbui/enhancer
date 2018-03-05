"""
https://arxiv.org/abs/1609.05158
https://github.com/tetrachrome/subpixel/blob/master/subpixel.py
"""
import numpy as np
import tensorflow as tf


def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0]  # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, b, a*r, r
    X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a * r, b * r, 1))


def phase_shift(X, r, color=False):
    if color:
        Xc = tf.split(X, 3, 3)
        X = tf.concat([_phase_shift(x, r) for x in Xc], 3)
    else:
        X = _phase_shift(X, r)
    return X


def PS(I, r):
    assert len(I.shape) == 3
    assert r > 0
    r = int(r)
    O = np.zeros((I.shape[0] * r, I.shape[1] * r, I.shape[2] / (r * 2)))
    for x in range(O.shape[0]):
        for y in range(O.shape[1]):
            for c in range(O.shape[2]):
                c += 1
                a = np.floor(x / r).astype("int")
                b = np.floor(y / r).astype("int")
                d = c * r * (y % r) + c * (x % r)
                print(a, b, d)
                O[x, y, c - 1] = I[a, b, d]
    return O


if __name__ == "__main__":
    with tf.Session() as sess:
        x = np.arange(2 * 16 * 16).reshape(2, 8, 8, 4)
        X = tf.placeholder("float32", shape=(2, 8, 8, 4), name="X")  # tf.Variable(x, name="X")
        Y = phase_shift(X, 2)
        y = sess.run(Y, feed_dict={X: x})
        print(y.shape)
        print(PS(x, 2))

        x2 = np.arange(2 * 3 * 16 * 16).reshape(2, 8, 8, 4 * 3)
        X2 = tf.placeholder("float32", shape=(2, 8, 8, 4 * 3), name="X")  # tf.Variable(x, name="X")
        Y2 = phase_shift(X2, 2, color=True)
        y2 = sess.run(Y2, feed_dict={X2: x2})
        print(y2.shape)
    # plt.imshow(y[0, :, :, 0], interpolation="none")
    # plt.show()
