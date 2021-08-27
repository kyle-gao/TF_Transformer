import tensorflow_datasets as tfds
from Transformer import *
import time
from preprocess import *


"""
The saved weights are for these specific parameters
num_layers = 4
d_model = 128
dense_dim = 256
num_heads = 8"""

num_layers = 4
d_model = 128
dense_dim = 256
num_heads = 8


max_len_en = 50
min_len_en = 10  # The transcript has many short lines indicating the date or speaker, which we should filter out.
max_len_fr = 50
min_len_fr = 10

EPOCHS = 1

eng_path = "Data/en.txt"
fr_path = "Data/fr.txt"

en_ds = tf.data.TextLineDataset(eng_path)
fr_ds = tf.data.TextLineDataset(fr_path)
ds = tf.data.Dataset.zip((fr_ds, en_ds))

"""tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
(en.numpy() for en, fr in ds), target_vocab_size=2**14)
tokenizer_en.save_to_file("tokenizer_en")

tokenizer_fr = tfds.features.text.SubwordTextEncoder.build_from_corpus(
  (fr.numpy() for en, fr in ds), target_vocab_size=2**14)
tokenizer_fr.save_to_file("tokenizer_fr")"""

#tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file("Data/tokenizer_en") #For tfds version <4.0
#tokenizer_fr = tfds.features.text.SubwordTextEncoder.load_from_file("Data/tokenizer_fr")

tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.load_from_file("Data/tokenizer_en") #For tfds version <4.0
tokenizer_fr = tfds.deprecated.text.SubwordTextEncoder.load_from_file("Data/tokenizer_fr")

en_vocab_size = tokenizer_en.vocab_size + 2
fr_vocab_size = tokenizer_fr.vocab_size + 2
transformer = Transformer(num_layers=num_layers, num_heads=num_heads, d_model=d_model, dense_dim=dense_dim,
                          in_vocab_size=fr_vocab_size, tar_vocab_size=en_vocab_size,
                          input_max_position=max_len_fr, target_max_position=max_len_en, rate=0.1)


def evaluate(question):

    start_token = [tokenizer_fr.vocab_size]
    end_token = [tokenizer_fr.vocab_size + 1]
    question = start_token + tokenizer_fr.encode(question) + end_token
    question = tf.expand_dims(question, 0)
    answer_in = [tokenizer_en.vocab_size]
    answer_in = tf.expand_dims(answer_in, 0)

    for i in range(max_len_fr):
        enc_padding_mask = padding_mask(question)
        dec_padding_mask = padding_mask(question)
        dec_forward_mask = forward_mask(answer_in)

        predictions = transformer(question, answer_in, training=False, enc_mask=enc_padding_mask,
                                  dec_forward_mask=dec_forward_mask, dec_padding_mask=dec_padding_mask)
        prediction = predictions[:, -1, :]  # select the last word to add to the outputs

        predicted_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)

        if predicted_id == tokenizer_en.vocab_size + 1:
            return tf.squeeze(answer_in, axis=0)
        predicted_id = tf.expand_dims(predicted_id, 0)
        answer_in = tf.concat([answer_in, predicted_id], axis=-1)

    return tf.squeeze(answer_in, axis=0)


def translate(sentence):
    result = np.array(evaluate(sentence))

    predicted_sentence = tokenizer_en.decode([i for i in result
                                              if tokenizer_en.vocab_size > i > 0])
    print('Input: {}'.format(sentence))
    print('Predicted answer: {}'.format(predicted_sentence))



def main():
    train_dataset = preprocess(ds, 64)
    train_dataset = train_dataset.take(10)

    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, warmup_steps=4000):
            super(CustomSchedule, self).__init__()

            self.d_model = d_model
            self.d_model = tf.cast(self.d_model, tf.float32)

            self.warmup_steps = warmup_steps

        def __call__(self, step):
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps ** -1.5)

            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    learning_rate = CustomSchedule(d_model, warmup_steps=4000)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    def masked_loss_fn(answer, prediction):
        mask = tf.math.logical_not(tf.math.equal(answer, 0))  # 0 at zeroes, 1 at non-zeroes since seq is padded
        # mask = tf.math.equal(answer, 0)
        mask = tf.cast(mask, tf.int32)
        loss_value = loss_fn(answer, prediction,
                             sample_weight=mask)  # set the zeros to zero weight, other values have weight of 1.

        return loss_value

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')


    signature = [tf.TensorSpec(shape=(None, max_len_fr), dtype=tf.int64),
                 tf.TensorSpec(shape=(None, max_len_en),
                               dtype=tf.int64), ]  # a bit faster if we specify the signature

    @tf.function(input_signature=signature)
    def train_step(question, answer):
        answer_in = answer[:, :-1]
        answer_tar = answer[:, 1:]

        enc_padding_mask = padding_mask(question)
        dec_padding_mask = padding_mask(question)
        dec_forward_mask = forward_mask(answer_in)

        with tf.GradientTape() as tape:
            predictions = transformer(question, answer_in, training=True, enc_mask=enc_padding_mask,
                                      dec_forward_mask=dec_forward_mask, dec_padding_mask=dec_padding_mask)
            loss = masked_loss_fn(answer_tar, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(answer_tar, predictions)

    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (question, answer)) in enumerate(train_dataset):
            train_step(question, answer)

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                            train_loss.result(),
                                                            train_accuracy.result()))

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
        translate("son honneur le president informe le senat que des senateurs attendent a la porte pour etre "
                        "presentes")


    #transformer.save_weights("transformer")
    transformer.load_weights("transformer")
    translate("son excellence le gouverneur general etant arrive au senat et ayant pris place sur le trone")


if __name__ == '__main__':
    main()
