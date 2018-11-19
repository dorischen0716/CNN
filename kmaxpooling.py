"""
Original Code from Internet
Modified to rank top k elements in dimension d
on 06/09/2018 by Doris Chen
"""


import tensorflow as tf

def kmaxpooling(inp_x,k,d=2):
    # d is the dimension to conduct kmaxpooling

    final_shape=inp_x.get_shape().as_list()
    len_truncated_d = final_shape[d]
    final_shape[d] = final_shape[-1]
    final_shape[-1] = k
    # tf.shape(x)[0] gives a scalar tensor with the variable batch size
    final_shape[0]=tf.shape(inp_x)[0]

    perm=[0, 1, 2,d]
    perm[d]=3
    inp_x =tf.transpose(inp_x, perm=perm)
    matrix_in = tf.reshape(inp_x, [-1,len_truncated_d])
    values, indices = tf.nn.top_k(matrix_in, k=k, sorted=False)
    shaped_out = tf.transpose(tf.reshape(tf.stack(values), final_shape), perm=perm)
    return shaped_out

    # final_shape = inp_x.get_shape().as_list()
    # d = final_shape[-2]
    # final_shape[-2] = final_shape[-1]
    # final_shape[-1] = k
    # # tf.shape(x)[0] gives a scalar tensor with the variable batch size
    # final_shape[0] = tf.shape(inp_x)[0]
    #
    # inp_x = tf.transpose(inp_x, perm=[0, 1, 3, 2])
    # matrix_in = tf.reshape(inp_x, [-1, d])
    # values, indices = tf.nn.top_k(matrix_in, k=k, sorted=False)
    # shaped_out = tf.transpose(tf.reshape(tf.stack(values), final_shape), perm=[0, 1, 3, 2])
    # return shaped_out

if __name__ == '__main__':
    with tf.Session() as sess:
        inp_x=tf.random_normal([2,4,1,3], mean=-1, stddev=4,seed=123)
        shaped_out=sess.run(kmaxpooling(inp_x,2))
        print(shaped_out)
