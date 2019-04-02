from ChatBot import ChatBot
import json
import numpy as np

with open('data/data.json', 'r') as myfile:
    data=myfile.read()
    # parse file
    obj = json.loads(data)

bot = ChatBot(obj)

# running tests
for seq_index in [344, 786, 233, 990, 1010, 539, 745, 984]:
    encoder_input_data = np.load('./tmp/encoder_input_data.npy')
    message = bot.dataset.content_recieved[seq_index]

    print('Input sentence:', message)

    dec_message = bot.send(message)

    print('-')
    print('Decoded sentence:', dec_message)


in_text = ""

print("Welcome to BOT")
print("To quit enter quit")

while True:
    in_text = input("You: ")
    if(in_text == "quit"):
        break
    print("The bot:", bot.send(in_text))
