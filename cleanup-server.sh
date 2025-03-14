#!/bin/bash

for pid in $(ls /proc | grep -E '^[0-9]+$'); do
    if ls -l /proc/$pid/fd 2>/dev/null | grep -q "socket:\[12345\]"; then
        echo "Killing PID $pid using port 8000"
        kill -9 $pid
    fi
done

