#!/bin/sh
# Pull the latest image from podman Hub
podman pull djana22/iris-fastapi:latest
# Stop and remove any existing container with the same name
( podman stop iris-fastapi 2>/dev/null || true )
( podman rm iris-fastapi 2>/dev/null || true )
# Run the container, mapping port 5000
podman run -d --name iris-fastapi -p 5000:5000 djana22/iris-fastapi:latest
