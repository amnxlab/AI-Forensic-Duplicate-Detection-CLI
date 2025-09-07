#!/bin/bash
# This script checks disk usage and sends an alert if over 80%

USAGE=$(df / | grep / | awk '{ print $5 }' | sed 's/%//g')

if [ $USAGE -gt 80 ]; then
  echo "ahmed space is critically low! Usage is at $USAGE%"
  # send email or log event here
else
  echo "Disk space usage is normal at $USAGE%"
fi
