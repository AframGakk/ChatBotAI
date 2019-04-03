# ChatBotAI
## Introduction
This project is a university project in artificial intelligence. The chatbot learns to talk from all your facebook 
conversations. The bot uses a sequence-to-sequence model with long-short term memory (LSTM) recurrant neural network 
encoder and decoder to calculate the probabilty sequence of the decoded message to send back to a user generated from
the message input.  

## How to use
The program works in several steps. The data pre-processing phase, the training phase and then the running phase. The 
project already comes with a small conversation package to try out but we recommend adding your own facebook messages 
into the data folder at the root of the project (more below).

#### Requirements
To run all aspects of the program you need the following
* python 3.6
* pip3

You also need to add these python packages with pip
* tensorflow
* keras
* numpy
* pandas

#### Data
If you choose to not train the bot with your own facebook convertations you can use our provided data and trained
models. You need to download our resources [here](https://www.dropbox.com/sh/v66gr1bjvoqwj92/AABfyC7-YDrhtIojFy_fQduqa?dl=0). 
The resources are quite large. When you downloaded the resources you need to place all the files in the data folder in the root of the project.

#### Quick start
To get started with the provided data and already trained models simpy execute the runChatBot.py. 
First open your terminal and navigate to root.
```
python3 runChatBot.py
```
Follow the instructions.

#### Training the bot
To start with you need to download all your facebook conversations. To get started follow 
[these](https://www.zapptales.com/en/download-facebook-messenger-chat-history-how-to/) instructions.
When you have your messages downloaded place the **messages** folder you got from facebook into the data
folder in the root of the project. When your set up execute the following.
1. Run the data pre-proccessing phase to strip data from your facebook messages with FCmain.py.
```
python3 FCmain.py
```
When the program finishes you have a new file in the data folder called data.json. This is used for training.
2. Start the training phase by executing trainChatBot.py. The program is set to only train 1 iteration (epoch) to
the autoencoder neural network for simulation only. If you truly want to train it well set the **epochs** variable
on line 9 in trainChatBot.py to 150 (or how many epochs you want) instead of 1.
```
python3 trainChatBot.py
```
Now wait........ and wait more.
3 Once the training is finished the program saves the autoencoder, encoder and decoder models to the data folder so
you don't need to train every time you want to run the bot. To get started run the runChatBot.py.
```
python3 runChatBot.py
```
Follow the instructions.

## Conclusion
In general the given data is not nearly enough or to complex to get the bot to learn syntax from the conversations. The
plan of attack is to make this program more dynamic in the way that you can add whatever conversation data is out there.
