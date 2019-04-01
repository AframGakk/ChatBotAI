#!/usr/bin/env bash

SERVER=13.58.43.216
DATA=

ssh -o StrictHostKeyChecking=no -i "~/.aws/ML.pem" ubuntu@13.58.43.216 "rm -r ~/bot/"
ssh -o StrictHostKeyChecking=no -i "~/.aws/ML.pem" ubuntu@13.58.43.216 "mkdir ~/bot"
ssh -o StrictHostKeyChecking=no -i "~/.aws/ML.pem" ubuntu@13.58.43.216 "mkdir ~/bot/scripts | mkdir ~/bot/data"

scp -o StrictHostKeyChecking=no -i "~/.aws/ML.pem" ./setup.sh ubuntu@13.58.43.216:~/bot/scripts

# Translator
scp -o StrictHostKeyChecking=no -i "~/.aws/ML.pem" ../TranslateBot.py ubuntu@13.58.43.216:~/bot/
scp -o StrictHostKeyChecking=no -i "~/.aws/ML.pem" ../TranslateBotTrainer.py ubuntu@13.58.43.216:~/bot/

# ChatBot
scp -o StrictHostKeyChecking=no -i "~/.aws/ML.pem" ../ChatBot.py ubuntu@13.58.43.216:~/bot/
scp -o StrictHostKeyChecking=no -i "~/.aws/ML.pem" ./ChatBotTrainer.py ubuntu@13.58.43.216:~/bot/

# Mains
scp -o StrictHostKeyChecking=no -i "~/.aws/ML.pem" ../train.py ubuntu@13.58.43.216:~/bot/data/
scp -o StrictHostKeyChecking=no -i "~/.aws/ML.pem" ../run.py ubuntu@13.58.43.216:~/bot/data/

# data files
scp -o StrictHostKeyChecking=no -i "~/.aws/ML.pem" ../data/fra.txt ubuntu@13.58.43.216:~/bot/data/

# Dependencies
scp -o StrictHostKeyChecking=no -i "~/.aws/ML.pem" ./SlackNotification.py ubuntu@$13.58.43.216:~/bot/data/


# RUNS
ssh -o StrictHostKeyChecking=no -i "~/.aws/ML.pem" ubuntu@$SERVER "~/bot/scripts/setup.sh"
ssh -o StrictHostKeyChecking=no -i "~/.aws/ML.pem" ubuntu@$SERVER "screen python3 ~/bot/TranslateBotTrainer.py"
ssh -o StrictHostKeyChecking=no -i "~/.aws/ML.pem" ubuntu@$SERVER "python3 ~/bot/train.py"




