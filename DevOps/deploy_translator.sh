#!/usr/bin/env bash

SERVER=18.188.78.150
DATA=

scp -o StrictHostKeyChecking=no -i "~/.aws/ML.pem" ./setup.sh ubuntu@18.188.78.150:~
scp -o StrictHostKeyChecking=no -i "~/.aws/ML.pem" ./run.sh ubuntu@18.188.78.150:~
ssh -o StrictHostKeyChecking=no -i "~/.aws/ML.pem" ubuntu@18.188.78.150 "./setup.sh"
ssh -o StrictHostKeyChecking=no -i "~/.aws/ML.pem" ubuntu@18.188.78.150 "./run.sh"





