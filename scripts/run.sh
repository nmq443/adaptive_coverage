#!/bin/bash

# Original method
# python3 scripts/run.py --env env0 --show_sensing_range True --controller hexagon --original_method True &

# PSO method
# python3 scripts/run.py --env env0 --show_sensing_range True --controller hexagon &

# Voronoi method
python3 scripts/run.py --env env2 --show_sensing_range True --controller voronoi --timestep 0.1
