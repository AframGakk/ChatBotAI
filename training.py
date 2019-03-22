import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# The datasets

dataset = pd.read_json('data/data.json')
#with open('data.json', 'rU') as f:
#    pd.read_json(data)

print("Dataset length: " + str(len(dataset)))
dataset = dataset.drop(columns=["timestamp_send", "timestamp_recieved"], axis=1)
dataset = dataset[:500]


latent_dim = 256  # Latent dimensionality of the encoding space.
batch_size = 64
epochs = 1 # 100

input_texts = []
target_texts = []
input_words = set()
target_words = set()
i = 0

for index, row in dataset.iterrows():
    input_text = row["content_recieved"].split()
    input_texts.append(row["content_recieved"])
    for word in input_text:
        input_words.add(word)
    target_text = row["content_sent"].split()
    target_texts.append(row["content_sent"])
    for word in target_text:
        target_words.add(word)

word_list = list(input_words)
target_list = list(target_words)
num_encoder_tokens = len(word_list)
num_decoder_tokens = len(target_list)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(target_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)


input_token_index = dict(
    [(char, i) for i, char in enumerate(word_list)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_list)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

print(len(decoder_input_data[0][0]))


for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, word in enumerate(input_text.split()):
        encoder_input_data[i, t, input_token_index[word]] = 1.
    for t, word in enumerate(target_text.split()):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        #decoder_input_data[i, t, target_token_index[word]] = 1.
        decoder_input_data[i, t] = target_token_index[word]
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[word]] = 1.


# Define an input sequence and process it.
encoder_inputs = Input(shape=(None,))
x = Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(x)

encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
dex = Embedding(num_decoder_tokens, latent_dim)
final_dex = dex(decoder_inputs)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(final_dex, initial_state=encoder_states)


decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)


# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


# Compile & run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)


# Save model
model.save('s2s.h5')
print("=================")
print("DONE")



