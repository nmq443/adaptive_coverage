@echo off

start cmd /k python scripts/run.py --env env6  --show_sensing_range --controller voronoi --save_video true
start cmd /k python scripts/run.py --env env6 --show_sensing_range True --controller hexagon --original_method false --save_video true
start cmd /k python scripts/run.py --env env6 --show_sensing_range True --controller hexagon --original_method true --save_video true