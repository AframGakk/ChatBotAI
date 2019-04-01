#!/usr/bin/env bash

# install python
if ! type python3 > /dev/null; then
    sudo apt-get install python3.6
fi

# install pip
if ! type pip3 > /dev/null; then
    sudo apt-get update

    sudo apt-get install python3-pip

    # pip install keras
    sudo pip install keras

    # pip install numpy
    sudo pip install numpy

    # pip install pandas
    sudo pip install pandas

    # pip install slackclient
    sudo pip install slacklient
fi



