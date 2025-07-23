#!/bin/bash

# Original method
python3 run.py --env env2 --show_sensing_range True --controller hexagon --original_method True &

# PSO method
python3 run.py --env env2 --show_sensing_range True --controller hexagon &

# Voronoi method
python3 run.py --env env2 --show_sensing_range True --controller voronoi &

# Wait to finish (optional)
wait
