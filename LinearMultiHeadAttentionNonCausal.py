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


class LinearMultiHeadAttentionNonCausal(tf.keras.layers.Layer):
    """Linear Attention Mechanism from Transformers are RNNs: Fast Autoregressive Transformers
    with Linear Attention by Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret.

    Uses linear feature maps f, to replace the softmax by a kernel k(x,y)->R+.
    so that f(x)*f(y) = k(x,y)

    This version lacks causal masking, is quite fast. Does not train well on word prediction tasks."""

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

    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size,len_q, dim_q)
        k = self.wk(k)  # (batch_size,len_v, dim_q)
        v = self.wv(v)  # (batch_size,len_v, dim_v)

        q = elu(self.split_heads(q, batch_size))  # (batch_size, seq_len_q, num_heads, depth_q) (m,l,h,d)
        k = elu(self.split_heads(k, batch_size))  # (batch_size,  seq_len_k, num_heads, depth_q) (m,j,h,d)
        v = self.split_heads(v, batch_size)  # (batch_size,  seq_len_v, num_heads, depth_v) (m,j,h,e)

        kv = tf.einsum("mjhd,mjhe->mdeh", k, v)  # (batch_size, depth_k, depth_v, seq_len_v)

        if mask is not None:  # padding mask is (m,j,1,1)
            # causal mask is (m,j,j,1) cannot be broadcast here
            k = k * mask

        # we contract k over the j axis and add an epsilon numerical stability.
        k_reduced = tf.math.reduce_sum(k, axis=1) + 1e-8

        z = 1 / (tf.einsum("mlhd,mhd->mlh", q, k_reduced))  # (batch_size, num_heads, seq_len_q)

        output = tf.einsum("mlhd,mdeh,mlh->mlhe", q, kv, z)  # (batch_size,len_q, heads, depth_v)
        output = tf.reshape(output, (batch_size, -1, self.num_heads * self.depth))  # (batch_size,len_q, d_model)

        return output
