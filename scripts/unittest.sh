#!/bin/bash

# If the input is all, run all tests.
if [ "$1" == "all" ]; then
    pytest tests
elif [ "$1" == "report" ]; then
    # Show the 20 slowest tests to identify performance bottlenecks.
    # Also reports the total execution time of the test suite.
    time pytest --durations=20 tests
elif [ -n "$1" ]; then
    # If the input file exists, test the input file.
    pytest $1
else
    # If nothing is input, test all.
    pytest tests
fi
