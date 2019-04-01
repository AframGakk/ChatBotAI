#!/usr/bin/env bash

SERVER=13.58.43.216
DATA=

scp -o StrictHostKeyChecking=no -i "~/.aws/ML.pem" ./setup.sh ubuntu@13.58.43.216:~
scp -o StrictHostKeyChecking=no -i "~/.aws/ML.pem" ./run.sh ubuntu@13.58.43.216:~
ssh -o StrictHostKeyChecking=no -i "~/.aws/ML.pem" ubuntu@13.58.43.216 "./setup.sh"
ssh -o StrictHostKeyChecking=no -i "~/.aws/ML.pem" ubuntu@13.58.43.216 "./run.sh"





