import json
from ChatBotTrainer import ChatBotTrainer
from TranslateBotTrainer import TranslateBotTrainer

with open('data/data.json', 'r') as myfile:
    data=myfile.read()
    # parse file
    obj = json.loads(data)

#trainer = ChatBotTrainer(obj, epochs=1)
trainer = TranslateBotTrainer('isl', epochs=1)
trainer.preProcess()
trainer.composeModel()
trainer.train()