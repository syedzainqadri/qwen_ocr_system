#!/bin/bash

# Flexible health check script that works with any port
# This script will be used in Docker health checks

# Get the port from environment variable or use default
PORT=${PORT:-3030}

# Try to curl the health endpoint
curl -f "http://localhost:${PORT}/health" || exit 1
