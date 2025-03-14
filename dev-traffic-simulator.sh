#!/bin/bash

i=1

while true
do
  # Check if the server is up on port 8000
  while ! nc -z 127.0.0.1 8000 2>/dev/null
  do
    echo "No server found on port 8000. Waiting 5 seconds before retrying..."
    sleep 5
  done

  # Now that the server is running, send your request
  prompt="How do I count to $i in French? (unique prompt #$i)"
  echo "Sending prompt: $prompt"

  curl http://127.0.0.1:8000/v1/completions \
    -X POST \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"llama-3.1-8b-engine\",
      \"prompt\": \"$prompt\",
      \"max_tokens\": 8000,
      \"temperature\": 0.8
    }"

  echo -e "\n"

  # Increment the counter to make each prompt unique
  i=$(( i + 1 ))

done

