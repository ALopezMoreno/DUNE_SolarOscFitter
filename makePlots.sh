#!/bin/bash

while read -r filename mystring; do
    # The rest of the line (after first space) goes into mystring
    python3 utils/plotOutput.py -e -o "$mystring" "$filename"
done < outlist.txt
