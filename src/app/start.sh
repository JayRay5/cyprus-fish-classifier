#!/bin/bash

echo "Starting the application..."
uvicorn src.app.api:app --host 0.0.0.0 --port 7860 