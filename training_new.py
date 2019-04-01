import pandas as pd
import numpy as np
import string
from string import digits
import matplotlib.pyplot as plt
#%matplotlib inline
import re
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model
from keras.utils import plot_model

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

embedding_size = 256
batch_size = 64
ephocs = 10

dataset = pd.read_json('data/data.json')
dataset = dataset.drop(columns=["timestamp_send", "timestamp_recieved"], axis=1)


# pre-processing clean up

dataset.content_recieved=dataset.content_recieved.apply(lambda x: x.lower())
dataset.content_sent=dataset.content_sent.apply(lambda x: x.lower())
# Take the length as 50
dataset.content_recieved=dataset.content_recieved.apply(lambda x: re.sub("'", '', x)).apply(lambda x: re.sub(",", ' COMMA', x))
dataset.content_sent=dataset.content_sent.apply(lambda x: re.sub("'", '', x)).apply(lambda x: re.sub(",", ' COMMA', x))
exclude = set(string.punctuation)
dataset.content_recieved=dataset.content_recieved.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
dataset.content_sent=dataset.content_sent.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
remove_digits = str.maketrans('', '', digits)
dataset.content_recieved=dataset.content_recieved.apply(lambda x: x.translate(remove_digits))
dataset.content_sent=dataset.content_sent.apply(lambda x: x.translate(remove_digits))

# TODO: kannski þarf að gera þetta eftir næsta skref
dataset.content_sent = dataset.content_sent.apply(lambda x : 'START_ '+ x + ' _END')

input_words = set()
target_words = set()

for eng in dataset.content_recieved:
    for word in eng.split():
        if word not in input_words:
            input_words.add(word)

for fr in dataset.content_sent:
    for word in fr.split():
        if word not in target_words:
            target_words.add(word)

# TODO: ef við þurfum að bæta start og end eftir á
# target_words = set(map(lambda x : 'START_ ' + x + ' _END', target_words))

lenght_list=[]
for l in dataset.content_recieved:
    lenght_list.append(len(l.split(' ')))
max_encoder_seq_length = np.max(lenght_list)


lenght_list=[]
for l in dataset.content_sent:
    lenght_list.append(len(l.split(' ')))
max_decoder_seq_length = np.max(lenght_list)


input_words_list = sorted(list(input_words))
target_words_list = sorted(list(target_words))
num_encoder_tokens = len(input_words)
num_decoder_tokens = len(target_words)
# del all_eng_words, all_french_words


input_token_index = dict(
    [(word, i) for i, word in enumerate(input_words_list)])
target_token_index = dict(
    [(word, i) for i, word in enumerate(target_words_list)])

encoder_input_data = np.zeros(
    (len(dataset.content_recieved), max_encoder_seq_length),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(dataset.content_sent), max_decoder_seq_length),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(dataset.content_sent), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(dataset.content_recieved, dataset.content_sent)):
    for t, word in enumerate(input_text.split()):
        encoder_input_data[i, t] = input_token_index[word]
    for t, word in enumerate(target_text.split()):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t] = target_token_index[word]
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[word]] = 1.


encoder_inputs = Input(shape=(None,))
en_x=  Embedding(num_encoder_tokens, embedding_size)(encoder_inputs)
encoder = LSTM(50, return_state=True)
encoder_outputs, state_h, state_c = encoder(en_x)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))

dex=  Embedding(num_decoder_tokens, embedding_size)

final_dex= dex(decoder_inputs)


decoder_lstm = LSTM(50, return_sequences=True, return_state=True)

decoder_outputs, _, _ = decoder_lstm(final_dex,
                                     initial_state=encoder_states)

decoder_dense = Dense(num_decoder_tokens, activation='softmax')

decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])


#TODO: model summary í file
#model.summary()


model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=ephocs,
          validation_split=0.2)

# Save model
model.save('saved_model.h5')

# TODO: encoder model summary í log skrá
encoder_model = Model(encoder_inputs, encoder_states)
#encoder_model.summary()


decoder_state_input_h = Input(shape=(50,))
decoder_state_input_c = Input(shape=(50,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

final_dex2= dex(decoder_inputs)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(final_dex2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index['START_']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += ' '+sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '_END' or
           len(decoded_sentence) > 52):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in [233, 245]:
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', dataset.content_sent[seq_index: seq_index + 1])
    print('Decoded sentence:', decoded_sentence)
