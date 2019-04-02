import pandas as pd
import numpy as np
import string
from string import digits
import re
from MessageParser import message_parse
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model
from keras.models import model_from_json


class ChatBotTrainer:


    def __init__(self, data, embedding_size = 256, batch_size = 64, epochs = 20):
        self.data = data
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.epochs = epochs

        self.cleanMessages()

        recieved = np.array(self.data[u'content_recieved'])
        sent = np.array(self.data[u'content_sent'])
        self.dataset = pd.DataFrame({'content_recieved': recieved, 'content_sent': sent})

        self.preProcess()
        self.composeModel()



    def cleanMessages(self):
        # clean up icelandic letters from JSON
        for idx, item in enumerate(self.data[u'content_sent']):
            self.data[u'content_sent'][idx] = message_parse(item)
        for idx, item in enumerate(self.data[u'content_recieved']):
            self.data[u'content_recieved'][idx] = message_parse(item)



    def preProcess(self):
        # pre-processing clean up

        self.dataset.content_recieved = self.dataset.content_recieved.apply(lambda x: str(x).lower())
        self.dataset.content_sent = self.dataset.content_sent.apply(lambda x: str(x).lower())

        # Take the length as 50
        self.dataset.content_recieved = self.dataset.content_recieved.apply(lambda x: re.sub("'", '', x)).apply(
            lambda x: re.sub(",", ' COMMA', x))
        self.dataset.content_sent = self.dataset.content_sent.apply(lambda x: re.sub("'", '', x)).apply(
            lambda x: re.sub(",", ' COMMA', x))

        exclude = set(string.punctuation)
        self.dataset.content_recieved = self.dataset.content_recieved.apply(
            lambda x: ''.join(ch for ch in x if ch not in exclude))
        self.dataset.content_sent = self.dataset.content_sent.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

        remove_digits = str.maketrans('', '', digits)
        self.dataset.content_recieved = self.dataset.content_recieved.apply(lambda x: x.translate(remove_digits))
        self.dataset.content_sent = self.dataset.content_sent.apply(lambda x: x.translate(remove_digits))

        self.dataset.content_sent = self.dataset.content_sent.apply(lambda x: 'START_ ' + x + ' _END')

        input_words = set()
        target_words = set()

        for recieved in self.dataset.content_recieved:
            for word in recieved.split():
                if word not in input_words:
                    input_words.add(word)

        for sent in self.dataset.content_sent:
            for word in sent.split():
                if word not in target_words:
                    target_words.add(word)

        lenght_list = []
        for l in self.dataset.content_recieved:
            lenght_list.append(len(l.split(' ')))
        max_encoder_seq_length = np.max(lenght_list)

        lenght_list = []
        for l in self.dataset.content_sent:
            lenght_list.append(len(l.split(' ')))
        max_decoder_seq_length = np.max(lenght_list)

        input_words_list = sorted(list(input_words))
        target_words_list = sorted(list(target_words))
        self.num_encoder_tokens = len(input_words)
        self.num_decoder_tokens = len(target_words)

        self.input_token_index = dict(
            [(word, i) for i, word in enumerate(input_words_list)])
        self.target_token_index = dict(
            [(word, i) for i, word in enumerate(target_words_list)])

        encoder_input_data = np.zeros(
            (len(self.dataset.content_recieved),max_encoder_seq_length),
            dtype='float32')
        decoder_input_data = np.zeros(
            (len(self.dataset.content_sent), max_decoder_seq_length),
            dtype='float32')
        decoder_target_data = np.zeros(
            (len(self.dataset.content_sent), max_decoder_seq_length, self.num_decoder_tokens),
            dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(self.dataset.content_recieved, self.dataset.content_sent)):
            for t, word in enumerate(input_text.split()):
                encoder_input_data[i, t] = self.input_token_index[word]

            for t, word in enumerate(target_text.split()):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t] = self.target_token_index[word]
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, self.target_token_index[word]] = 1.
        np.save('./tmp/encoder_input_data', encoder_input_data)
        np.save('./tmp/decoder_input_data', decoder_input_data)
        np.save('./tmp/decoder_target_data', decoder_target_data)



    def composeModel(self):
        self.encoder_inputs = Input(shape=(None,))
        en_x = Embedding(self.num_encoder_tokens, self.embedding_size)(self.encoder_inputs)
        encoder = LSTM(50, return_state=True)
        encoder_outputs, state_h, state_c = encoder(en_x)
        # We discard `encoder_outputs` and only keep the states.
        self.encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        self.decoder_inputs = Input(shape=(None,))

        self.dex = Embedding(self.num_decoder_tokens, self.embedding_size)

        final_dex = self.dex(self.decoder_inputs)

        self.decoder_lstm = LSTM(50, return_sequences=True, return_state=True)

        decoder_outputs, _, _ = self.decoder_lstm(final_dex,
                                             initial_state=self.encoder_states)

        self.decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')

        decoder_outputs = self.decoder_dense(decoder_outputs)

        self.model = Model([self.encoder_inputs, self.decoder_inputs], decoder_outputs)

        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])



    def train(self):

        if not self.model:
            self.composeModel()

        encoder_input_data = np.load('./tmp/encoder_input_data.npy')
        decoder_input_data = np.load('./tmp/decoder_input_data.npy')
        decoder_target_data = np.load('./tmp/decoder_target_data.npy')
        try:
            self.model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                      batch_size=self.batch_size,
                      epochs=self.epochs,
                      validation_split=0.08)
        except:
            print("There was a problem with the model fitting")

        # print the summary for the autoencoder
        self.modelSummary("autoencoder", self.model)

        # create the sampling model
        self.samplingModel()



    def samplingModel(self):
        if not self.model:
            print("Please train the model with .train()")
            return

        # TODO: encoder model summary í log skrá
        self.encoder_model = Model(self.encoder_inputs, self.encoder_states)

        decoder_state_input_h = Input(shape=(50,))
        decoder_state_input_c = Input(shape=(50,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        final_dex2 = self.dex(self.decoder_inputs)

        decoder_outputs2, state_h2, state_c2 = self.decoder_lstm(final_dex2, initial_state=decoder_states_inputs)
        decoder_states2 = [state_h2, state_c2]
        decoder_outputs2 = self.decoder_dense(decoder_outputs2)
        self.decoder_model = Model(
            [self.decoder_inputs] + decoder_states_inputs,
            [decoder_outputs2] + decoder_states2)

        # Reverse-lookup token index to decode sequences back to
        # something readable.
        self.reverse_input_char_index = dict(
            (i, char) for char, i in self.input_token_index.items())
        self.reverse_target_char_index = dict(
            (i, char) for char, i in self.target_token_index.items())

        print("Decoder setup DONE")

        #print summary for the encoder
        self.modelSummary("Encoder", self.encoder_model)

        #print summary for the decoder
        self.modelSummary("Decoder", self.decoder_model)



    def decoder(self, input_seq):
        if not self.model:
            print("Please train the model with .train()")
            return

        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0] = self.target_token_index['START_']

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_target_char_index[sampled_token_index]
            decoded_sentence += ' ' + sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '_END' or
                    len(decoded_sentence) > 11):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # Update states
            states_value = [h, c]

        return decoded_sentence



    def test(self):
        encoder_input_data = np.load('./tmp/encoder_input_data.npy')
        for seq_index in [344, 786, 233, 990, 1010, 539, 745, 984]:
            input_seq = encoder_input_data[seq_index: seq_index + 1]
            decoded_sentence = self.decoder(input_seq)
            print('-')
            print('Input sentence:', self.dataset.content_recieved[seq_index])
            print('Decoded sentence:', decoded_sentence)



    def printAllRecieved(self):
        for item in self.dataset.content_recieved:
            print(item)



    def saveModels(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("./data/model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("./data/model.h5")
        print("Saved model to disk")

        # serialize model to JSON
        model_json = self.encoder_model.to_json()
        with open("./data/encoder_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.encoder_model.save_weights("./data/encoder_model.h5")
        print("Saved model to disk")

        # serialize model to JSON
        model_json = self.decoder_model.to_json()
        with open("./data/decoder_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.decoder_model.save_weights("./data/decoder_model.h5")
        print("Saved model to disk")



    def loadModel(self):
        # load json and create model
        json_file = open('./data/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("./data/model.h5")
        print("Loaded model from disk")



    def modelSummary(self, model_name,model):
        if(model):
            print("Summary for model: ", model_name)
            print(model.summary())
        else:
            print("No model has been trained")
