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
from MultiHeadAttention import *
from helperfunctions import *


class EncoderLayer(tf.keras.layers.Layer):
    """The EncoderLayer consists of one MultiHeadAttention layer connected to a FeedForward layer,
    each of these 2 layers have a residual connection."""

    def __init__(self, num_heads, d_model, dense_dim, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.dense = tf.keras.Sequential([tf.keras.layers.Dense(dense_dim, activation='relu'),
                                          tf.keras.layers.Dense(d_model)])

        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training, mask):
        out_attention = self.attention(x, x, x, mask)  # (batch_size,seq_len,d_model)
        out_attention = self.dropout1(out_attention, training=training)
        out1 = self.norm1(x + out_attention)  # residual connection (batch_size,seq_len,d_model)

        out_dense = self.dense(out1)  # (batch_size,seq_len,d_model)
        out2 = self.norm2(out1 + out_dense)  # residual conenction (batch_size,seq_len,d_model)
        return out2


class Encoder(tf.keras.layers.Layer):
    """The Encoder consists of EncoderLayer"""

    def __init__(self, num_layers, num_heads, d_model, dense_dim,
                 vocab_size, max_encoding_position, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.positional_encoding = positional_encoding(max_encoding_position, d_model)
        self.encoding_layers = [EncoderLayer(num_heads, d_model, dense_dim, dropout) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training, mask=None):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)  # (batch_size,input_len,d_model)
        x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.positional_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.encoding_layers[i](x, training, mask)  # (batch_size, input_seq_len, d_model)

        return x
