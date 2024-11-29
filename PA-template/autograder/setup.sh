#!/usr/bin/env bash

apt-get -y install openjdk-11-jdk
apt-get install -y python3 python3-pip python3-dev

pip3 install -r /autograder/source/requirements.txt
# just to be VERY safe we didn't add sym link to solution/submission.py
rm -rf /autograder/source/submission.py
