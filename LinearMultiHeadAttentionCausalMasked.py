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
from helperfunctions import elu


class MultiHeadAttentionCausalMasked(tf.keras.layers.Layer):
    """LinearAttention Mechanism from Transformers are RNNs: Fast Autoregressive Transformers
    with Linear Attention by Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret.

    Uses linear feature maps f, to replace the softmax by a kernel k(x,y)->R+.
    so that f(x)*f(y) = k(x,y)

    The authors use the elu feature map.

    This version has causal i.e. forward masking. This cannot be implemented in the usual way due to the
    Q*K term not existing in isolation in this Linear Attention.
    I have implemented it in clumsy way which makes this slower than the usual softmax attention by quite a bit.

    Tne authors of the paper implemented causal attention via a triangular tensor product (and its back prop) in c++.

    I have implemented it in clumsy way which makes this slower than the usual softmax attention by quite a bit
    by introducing an intermediate step with the dimensions of the Q*K product.
    """

    """NOTE!!! Due to change of dimensionality of the tensor in the intermediate step, 
    a different masking scheme must be used. Use helperfunctions.forward_mask5() and 
    helper.functions.padding_mask5() on any transformer which calls this attention layer."""

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
        x -- A tokenized sequence (batch_size,seq_len,d_model)

        Returns:
        A tokenized sequence with dimensions (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return x

    def call(self, q, k, v, mask=None, eps=1e-8):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size,len_q, dim_q)
        k = self.wk(k)  # (batch_size,len_v, dim_q)
        v = self.wv(v)  # (batch_size,len_v, dim_v)

        q = elu(self.split_heads(q, batch_size))  # (batch_size, seq_len_q, num_heads, depth_q) (m,l,h,d)
        k = elu(self.split_heads(k, batch_size))  # (batch_size,  seq_len_v, num_heads, depth_q) (m,j,h,d)
        v = self.split_heads(v, batch_size)  # (batch_size,  seq_len_v, num_heads, depth_v) (m,j,h,e)

        k_reduced = tf.math.reduce_sum(k, axis=1) + 1e-8

        z = 1 / (tf.einsum("mlhd,mhd->mlh", q, k_reduced))  # (batch_size, num_heads, seq_len_q)

        output = tf.einsum("mjhd,mjhe->mjehd", k, v)  # (batch_size, len_v, depth_q, num_heads, depth_v)

        output = tf.einsum("mlhd,mjehd,mlh->mjlhe", q, output, z)  # (batch_size, len_q, len_v, num_heads, depth_v)

        if mask is not None:
            output = output * mask  # Mask must broadcast to j and l axis correctly

        output = tf.einsum("mjlhe->mlhe", output)

        output = tf.reshape(output, (batch_size, -1, self.num_heads * self.depth))  # (batch_size,len_q, d_model)
        return output  # (m,l,h*e)