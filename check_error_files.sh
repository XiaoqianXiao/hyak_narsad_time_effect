#!/bin/bash

# Print filename if last line does NOT contain "Group-level analysis pipeline completed successfully"

for file in *.err; do
    if [[ -f "$file" ]]; then
        last_line=$(tail -n 1 "$file" 2>/dev/null)
        if [[ "$last_line" != *"completed successfully"* ]]; then
            echo "$file"
        fi
    fi
done

for file in *flameo*.err; do
    if [[ -f "$file" ]]; then
        last_line=$(tail -n 1 "$file" 2>/dev/null)
        if [[ "$last_line" != *"analysis pipeline completed successfully"* ]]; then
            echo "$file"
        fi
    fi
done

for file in *randomise*.err; do
    if [[ -f "$file" ]]; then
        last_line=$(tail -n 1 "$file" 2>/dev/null)
        if [[ "$last_line" != *"analysis pipeline completed successfully"* ]]; then
            echo "$file"
        fi
    fi
done

for file in *.err; do
    if [[ -f "$file" ]]; then
        last_line=$(tail -n 1 "$file" 2>/dev/null)
        if [[ "$last_line" != *"completed successfully"* ]]; then
            echo "$file"
        fi
    fi
done


for file in *flameo*.err; do
    if [[ -f "$file" ]]; then
        last_line=$(tail -n 1 "$file" 2>/dev/null)
        if [[ "$last_line" != *"analysis pipeline completed successfully"* ]]; then
            echo "$file"
        fi
    fi
done

for file in *flameo*.err; do
    if [[ -f "$file" ]]; then
        last_line=$(tail -n 1 "$file" 2>/dev/null)
        if [[ "$last_line" == *"ERROR"* ]]; then
            echo "$file"
        fi
    fi
done

for file in *.err; do
    if [[ -f "$file" ]]; then
        last_line=$(tail -n 1 "$file" 2>/dev/null)
        if [[ "$last_line" == *"ERROR"* ]]; then
            echo "$file"
        fi
    fi
done

for file in *.err; do
    if [[ -f "$file" ]]; then
        last_line=$(tail -n 1 "$file" 2>/dev/null)
        if [[ "$last_line" != *"ERROR"* ]]; then
            echo "$file"
        fi
    fi
done