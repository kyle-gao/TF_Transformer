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

from MultiHeadAttention import *
import tensorflow as tf
from helperfunctions import *


class DecoderLayer(tf.keras.layers.Layer):
    """A decoder layers consists of two MultiHeadAttention, one for the Decoder input, one from Encoder output"""
    def __init__(self, num_heads, d_model, dense_dim, dropout=0.1):
        super().__init__()

        self.attention1 = MultiHeadAttention(d_model, num_heads)
        self.attention2 = MultiHeadAttention(d_model, num_heads)

        self.dense = tf.keras.Sequential([tf.keras.layers.Dense(dense_dim, activation='relu'),
                                          tf.keras.layers.Dense(d_model)])

        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.norm3 = tf.keras.layers.LayerNormalization()

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.dropout3 = tf.keras.layers.Dropout(dropout)

    def call(self, encoder_out, x, training, forward_mask, padding_mask):

        out_attention1 = self.attention1(x, x, x,
                                         forward_mask)  # (batch_size, seq_len_answer, d_model) -> The return seq_len is the same as that of the first argument of the call.
        out_attention1 = self.dropout1(out_attention1, training=training)
        out1 = self.norm1(x + out_attention1)  # residual connection (batch_size, seq_len_answer, d_model)

        out_attention2 = self.attention2(out1, encoder_out, encoder_out,
                                         padding_mask)  # (batch_size, seq_len_answer, d_model)
        out_attention2 = self.dropout2(out_attention2, training=training)
        out2 = self.norm2(out1 + out_attention2)

        out_dense = self.dense(out2)
        out_dense = self.dropout3(out_dense + out2)

        return out_dense


class Decoder(tf.keras.layers.Layer):
    """The Decoder consists of multiple DecoderLayer"""
    def __init__(self, num_layers, num_heads, d_model, dense_dim,
                 vocab_size, max_encoding_position, dropout=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.positional_encoding = positional_encoding(max_encoding_position, d_model)
        self.decoder_layers = [DecoderLayer(num_heads, d_model, dense_dim, dropout) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, encoder_out, x, training, forward_mask=None, padding_mask=None):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)  # (batch_size,input_len,d_model)
        x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.positional_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.decoder_layers[i](encoder_out, x, training, forward_mask,
                                       padding_mask)  # (batch_size, input_seq_len, d_model)
        return x