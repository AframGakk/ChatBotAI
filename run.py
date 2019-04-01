from ChatBot import ChatBot
import json

with open('data/data.json', 'r') as myfile:
    data=myfile.read()
    # parse file
    obj = json.loads(data)

bot = ChatBot(obj)
bot.setupDecoder()

in_text = ""

print("Welcome to BOT")
print("To quit enter quit")

while in_text is not "quit":
    in_text = input()
    print(bot.decoder(in_text))