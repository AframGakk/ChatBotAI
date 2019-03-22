from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.load_weights("seq2seq_eng-ger.h5")