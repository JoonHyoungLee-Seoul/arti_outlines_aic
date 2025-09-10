#!/bin/bash

echo "Stopping virtual display server..."

# Kill Xvfb processes
pkill -f Xvfb

# Clean up PID file
if [ -f /tmp/xvfb.pid ]; then
    PID=$(cat /tmp/xvfb.pid)
    kill $PID 2>/dev/null
    rm /tmp/xvfb.pid
    echo "Cleaned up Xvfb PID: $PID"
fi

# Unset DISPLAY
unset DISPLAY

echo "Virtual display stopped and cleaned up!"