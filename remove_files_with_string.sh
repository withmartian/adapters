#!/bin/bash

# Check if a string argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <string>"
    exit 1
fi

# Assign the first argument to a variable
search_string=$1

# Optional: Specify directory or default to current
directory="."

# Find and remove files containing the string in their name
find "$directory" -type f -name "*$search_string*" -exec rm -v {} +

# Completion message
echo "Files containing '$search_string' have been removed from $directory."
