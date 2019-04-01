from ChatBot import ChatBot
import json

with open('data/data_villi.json', 'r') as myfile:
    data=myfile.read()
    # parse file
    obj = json.loads(data)

bot = ChatBot(obj)
bot.setupDecoder()

# running tests
for seq_index in [344, 786, 233, 990, 1010, 539, 745, 984]:
    input_seq = bot.encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = bot.decoder(input_seq)
    print('-')
    print('Input sentence:', bot.dataset.content_recieved[seq_index])
    print('Decoded sentence:', decoded_sentence)

'''
in_text = ""

print("Welcome to BOT")
print("To quit enter quit")

while in_text is not "quit":
    in_text = input()
    print(bot.decoder(in_text))
'''