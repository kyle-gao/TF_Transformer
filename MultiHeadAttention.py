"""
Copyright 2020 Yi Lin(Kyle) Gao
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License."""

import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):

    """Implemented with tf.einsum(), is faster than using tf.transpose() with tf.matmul()"""

    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads,depth)

        Arguments:
        x -- A tokenized sequence (batch_size, seq_len, d_model)

        Returns:
        A tokenized sequence with dimensions (batch_size, seq_len, num_heads, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))

        return x

    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # (batch_size,len_q, dim_q)
        k = self.wk(k)  # (batch_size,len_v, dim_q)
        v = self.wv(v)  # (batch_size,len_v, dim_v)

        q = self.split_heads(q, batch_size)  # (batch_size, len_q, num_heads, depth_q) (m,l,h,d)
        k = self.split_heads(k, batch_size)  # (batch_size, len_v, num_heads, depth_q) (m,j,h,d)
        v = self.split_heads(v, batch_size)  # (batch_size, len_v, num_heads, depth_v) (m,j,h,e)

        qk = tf.einsum("mlhd,mjhd->mljh", q, k)  # (batch_size, len_q, len_v, num_heads) (m,l,j,h)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        qk = qk / tf.math.sqrt(dk)

        if mask is not None:
            qk = qk - mask*1e9 # We are using a multiplicative mask

        qk = tf.nn.softmax(qk, axis=-2)  # (batch_size,len_q,len_v, num_heads) (m,l,j,h)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        qk = qk / tf.math.sqrt(dk)

        output = tf.einsum("mljh, mjhe -> mlhe", qk, v)  # (batch_size,len_q, heads, depth_v)
        output = tf.reshape(output, (batch_size, -1, self.num_heads * self.depth))  # (batch_size,len_q, d_model)

        return self.dense(output)
