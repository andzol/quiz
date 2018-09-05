#!/bin/bash

set -e 
gzip -c ./tt_dataset/pretrained.xml >model.gz
xxd -i model.gz >model.h && rm -f model.gz
echo "$(ls -la model.h)"
