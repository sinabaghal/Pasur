#!/bin/bash

# Define your list of indices
mylist=(10 112 253)

# Compile each LaTeX file
for i in "${mylist[@]}"; do
    pdflatex "games/game_$i.txt"
done