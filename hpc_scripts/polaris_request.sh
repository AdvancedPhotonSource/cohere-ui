#!/bin/bash

CONF=$1
DATA_FILE=$2
NRECS=$3
IS_GA=$4

NNODES=$((python estimate.py $CONF $DATA_FILE $NRECS $IS_GA) 2>&1)

sh request_q.sh $NNODES
