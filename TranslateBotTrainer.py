import pandas as pd
import numpy as np
import string
from string import digits
import re
from MessageParser import message_parse
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model
from SlackNotification import SlackNotify

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


class TranslateBotTrainer:

    def __init__(self, data, embedding_size = 256, batch_size = 64, epochs = 20):
        self.data = data
        self.datafile = "./data/" + data + ".txt"
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.epochs = epochs

        self.dataset = pd.read_table(self.datafile, names=['eng', 'isl'])

    def cleanMessages(self):
        # clean up icelandic letters from JSON
        for key in self.data[u'content_sent']:
            self.data[u'content_sent'][key] = message_parse(self.data[u'content_sent'][key])

        for key in self.data[u'content_recieved']:
            self.data[u'content_recieved'][key] = message_parse(self.data[u'content_recieved'][key])


    def preProcess(self):
        # pre-processing clean up

        self.dataset.eng = self.dataset.eng.apply(lambda x: str(x).lower())
        self.dataset.isl = self.dataset.isl.apply(lambda x: str(x).lower())

        # Take the length as 50
        self.dataset.eng = self.dataset.eng.apply(lambda x: re.sub("'", '', x)).apply(
            lambda x: re.sub(",", ' COMMA', x))
        self.dataset.isl = self.dataset.isl.apply(lambda x: re.sub("'", '', x)).apply(
            lambda x: re.sub(",", ' COMMA', x))

        exclude = set(string.punctuation)
        self.dataset.eng = self.dataset.eng.apply(
            lambda x: ''.join(ch for ch in x if ch not in exclude))
        self.dataset.isl = self.dataset.isl.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

        remove_digits = str.maketrans('', '', digits)
        self.dataset.eng = self.dataset.eng.apply(lambda x: x.translate(remove_digits))
        self.dataset.isl = self.dataset.isl.apply(lambda x: x.translate(remove_digits))

        self.dataset.isl = self.dataset.isl.apply(lambda x: 'START_ ' + x + ' _END')

        input_words = set()
        target_words = set()

        for recieved in self.dataset.eng:
            for word in recieved.split():
                if word not in input_words:
                    input_words.add(word)

        for sent in self.dataset.isl:
            for word in sent.split():
                if word not in target_words:
                    target_words.add(word)

        lenght_list = []
        for l in self.dataset.eng:
            lenght_list.append(len(l.split(' ')))
        max_encoder_seq_length = np.max(lenght_list)

        lenght_list = []
        for l in self.dataset.isl:
            lenght_list.append(len(l.split(' ')))
        max_decoder_seq_length = np.max(lenght_list)

        input_words_list = sorted(list(input_words))
        target_words_list = sorted(list(target_words))
        self.num_encoder_tokens = len(input_words)
        self.num_decoder_tokens = len(target_words)
        # del all_eng_words, all_french_words

        input_token_index = dict(
            [(word, i) for i, word in enumerate(input_words_list)])
        target_token_index = dict(
            [(word, i) for i, word in enumerate(target_words_list)])

        self.encoder_input_data = np.zeros(
            (len(self.dataset.eng),max_encoder_seq_length),
            dtype='float32')
        self.decoder_input_data = np.zeros(
            (len(self.dataset.isl), max_decoder_seq_length),
            dtype='float32')
        self.decoder_target_data = np.zeros(
            (len(self.dataset.isl), max_decoder_seq_length, self.num_decoder_tokens),
            dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(self.dataset.eng, self.dataset.isl)):
            for t, word in enumerate(input_text.split()):
                self.encoder_input_data[i, t] = input_token_index[word]
            for t, word in enumerate(target_text.split()):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                self.decoder_input_data[i, t] = target_token_index[word]
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    self.decoder_target_data[i, t - 1, target_token_index[word]] = 1.


    def composeModel(self):
        self.encoder_inputs = Input(shape=(None,))
        en_x = Embedding(self.num_encoder_tokens, self.embedding_size)(self.encoder_inputs)
        encoder = LSTM(50, return_state=True)
        encoder_outputs, state_h, state_c = encoder(en_x)
        # We discard `encoder_outputs` and only keep the states.
        self.encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        self.decoder_inputs = Input(shape=(None,))

        dex = Embedding(self.num_decoder_tokens, self.embedding_size)

        final_dex = dex(self.decoder_inputs)

        decoder_lstm = LSTM(50, return_sequences=True, return_state=True)

        decoder_outputs, _, _ = decoder_lstm(final_dex,
                                             initial_state=self.encoder_states)

        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')

        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = Model([self.encoder_inputs, self.decoder_inputs], decoder_outputs)

        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])


    def train(self):

        if not self.model:
            self.composeModel()

        self.model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  validation_split=0.1)

        # Save model
        self.saveModel()

        self.encoder_model = Model(self.encoder_inputs, self.encoder_states)

        SlackNotify("The bot has finished training", "chat-bot")


    def saveModel(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("./data/" + self.data + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("./data/" + self.data + ".h5")
        print("Saved model to disk")


    def summary(self):
        if(self.model):
            print(self.model.summary())
        else:
            print("No model has been trained")
