@echo off

start cmd /k python scripts/run.py --env env0  --show_sensing_range --controller voronoi
start cmd /k python scripts/run.py --env env0 --show_sensing_range True --controller hexagon --original_method false
start cmd /k python scripts/run.py --env env0 --show_sensing_range True --controller hexagon --original_method true