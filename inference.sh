#!/bin/bash

set -e

echo "--- Starting Submission Pipeline ---"
echo "-- Building Vector DB -- "
python3 build_db.py


echo "Running prediction..."
python3 predict.py

echo "--- Pipeline Finished Successfully ---"