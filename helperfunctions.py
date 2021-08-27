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

import numpy as np
import tensorflow as tf


def elu(z):
    """elu feature map used by Katharopoulos et al."""
    return tf.nn.elu(z) + 1


def positional_encoding(pos, d_model):
    """
    :param pos: int max position
    :param d_model: dimension of the model
    :return: (1,pos,d_model) array of sinusoidal positional encoding
    """
    pos_enc = np.zeros((1, pos, d_model))
    for p in range(pos):
        for i in range(d_model // 2):
            angles = p / np.power(10000, (2 * i) / np.float32(d_model))
            pos_enc[:, p, 2 * i] = np.sin(angles)
            pos_enc[:, p, 2 * i + 1] = np.cos(angles)
        if d_model % 2 == 1:
            # if d_model is odd loop doesn't hit last even index
            angles = p / np.power(10000, (2 * d_model) / np.float32(d_model))
            pos_enc[:, p, d_model - 1] = np.sin(angles)
    return tf.cast(pos_enc, tf.float32)


def padding_mask(seq):
    # Returns (batch, seq_len, 1, 1) tensor with 1's where the sequence is padded, 0 where it is not

    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return mask[:, tf.newaxis, :,  tf.newaxis]  # (batch, 1, seq_len, 1) m l j h  <- j gets masked


def forward_mask(seq):
    """
    Calculates a combined forward mask and padding mask for a batch of sequences
    :param seq: (batch,seq_len) a batch of sequences
    :return:  a combined look_ahead_mask (upper triangular 1s)
    and padding mask (batch, seq_len, seq_len, 1)
    """
    seq_len = tf.shape(seq)[1]

    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    look_ahead_mask = look_ahead_mask[tf.newaxis, :, :, tf.newaxis]  # (batch, seq_len, seq_len, 1)

    padded_mask = padding_mask(seq)

    # return padded_mask * look_ahead_mask  # (batch, seq_len, seq_len, 1)
    return tf.maximum(padded_mask, look_ahead_mask)