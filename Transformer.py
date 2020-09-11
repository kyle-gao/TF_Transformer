import tensorflow as tf
from Encoder import *
from Decoder import *


class Transformer(tf.keras.Model):

    def __init__(self, num_layers, num_heads, d_model, dense_dim, in_vocab_size, tar_vocab_size,
                 input_max_position, target_max_position, rate=0.1):
        super().__init__()

        self.encoder = Encoder(num_layers, num_heads, d_model, dense_dim,
                               in_vocab_size, max_encoding_position=input_max_position, dropout=0.1)

        self.decoder = Decoder(num_layers, num_heads, d_model, dense_dim,
                               tar_vocab_size, max_encoding_position=target_max_position, dropout=0.1)

        self.dense = tf.keras.layers.Dense(tar_vocab_size)

    def call(self, input, target, training=False, enc_mask=None, dec_forward_mask=None, dec_padding_mask=None):
        out_encoder = self.encoder(input, training=training, mask=enc_mask)

        out_decoder = self.decoder(out_encoder, target, training=training, forward_mask=dec_forward_mask,
                                   padding_mask=dec_padding_mask)

        out = self.dense(out_decoder)

        return out
