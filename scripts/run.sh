#!/bin/bash

# First command (runs in background)
python3 main.py --env env2 --show_sensing_range True --controller hexagon --original_method False &

# Second command (runs in background)
python3 main.py --env env2 --show_sensing_range True --controller voronoi &

# Wait for both to finish (optional)
wait
