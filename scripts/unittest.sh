#!/bin/bash

# If the input is all, run all tests.
if [ "$1" == "all" ]; then
    pytest tests
elif [ -n "$1" ]; then
    # If the input file exists, test the input file.
    pytest $1
else
    # If nothing is input, test all.
    pytest tests
fi