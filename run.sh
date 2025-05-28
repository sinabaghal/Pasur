#!/bin/bash

# Define your list of indices
mylist=()

for filepath in games/game_*.txt; do
    # Extract filename from path, e.g. game_10.txt
    filename=$(basename "$filepath")
    
    # Extract number between underscore and .txt
    id=${filename#game_}   # removes 'game_'
    id=${id%.txt}          # removes '.txt'
    
    mylist+=("$id")
done

Compile each LaTeX file
for i in "${mylist[@]}"; do
    pdflatex "games\game_$i.txt"
done


files=""

# Build the list of PDF files
for i in "${mylist[@]}"; do
    files+="game_${i}.pdf "
done

# Run pdfunite on all files and create combined.pdf
pdfunite $files combined.pdf
find . -maxdepth 1 -type f -name "game_*" -exec rm {} \;
