#!/bin/bash
# startup.sh - Script to ensure proper permissions and set up environment before starting MSP4GROWTH

# Echo commands for debugging
set -x

# Create required directories with correct permissions
mkdir -p ./data ./results ./logs
chmod 777 ./logs  # Ensure the logs directory is writable by container

# Rebuild and restart the containers
docker-compose down
docker-compose build
docker-compose up -d

# Show logs to check if application starts correctly
echo "Waiting for container to start..."
sleep 5
docker-compose logs