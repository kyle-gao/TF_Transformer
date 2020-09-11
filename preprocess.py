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


from main import *


def encode(fr, eng):
    # Adds start token (tokenizer.vocab_size) and end token (tokenizer.vocab_size + 1) to (question,answer)
    question = [tokenizer_fr.vocab_size] + tokenizer_fr.encode(fr.numpy()) + [tokenizer_fr.vocab_size + 1]
    answer = [tokenizer_en.vocab_size] + tokenizer_en.encode(eng.numpy()) + [tokenizer_en.vocab_size + 1]

    return question, answer


def tf_interleave_encode(question, answer):
    # We have to wrap encode in a tf.py_function() and return a Dataset so we can use Dataset.interleave()
    question, answer = tf.py_function(encode, [question, answer], [tf.int64, tf.int64])
    question.set_shape([None])
    answer.set_shape([None])

    return tf.data.Dataset.from_tensors((question, answer))


def filter_max_length(x, y, max_length_question=max_len_fr, max_length_answer=max_len_en):
    return tf.logical_and(tf.size(x) <= max_length_question,
                          tf.size(y) <= max_length_answer)


def filter_min_length(x, y, min_len_question=min_len_fr, min_len_answer=min_len_en):
    return tf.logical_and(tf.size(x) >= min_len_question,
                          tf.size(y) >= min_len_answer)


def preprocess(dataset, batch_size, pad_len_question=max_len_fr, pad_length_answer=max_len_en):
    dataset = dataset.cache()
    # dataset = dataset.map(tf_encode)
    dataset = dataset.interleave(tf_interleave_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.filter(filter_max_length)
    dataset = dataset.filter(filter_min_length)
    dataset = dataset.shuffle(10000)

    dataset = dataset.padded_batch(batch_size, drop_remainder=True,
                                   padded_shapes=([pad_len_question], [pad_length_answer]))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
