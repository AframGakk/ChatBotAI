from TranslateBotTrainer import TranslateBotTrainer

trainer = TranslateBotTrainer('isl', epochs=20)
trainer.preProcess()
trainer.composeModel()
trainer.train()