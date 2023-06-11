#!/bin/bash

word="$1"

command="gcc -Wall -O3 ${word} -lm && ./a.out"

eval "$command"
