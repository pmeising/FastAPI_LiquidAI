#!/bin/bash

echo "Starting LiquidAI FastAPI Service..."
echo

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

echo "Building and starting the service..."
docker-compose up --build
