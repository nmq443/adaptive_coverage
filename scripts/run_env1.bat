@echo off

start cmd /c python scripts/run.py --env env1 --show_sensing_range --controller voronoi
start cmd /c python scripts/run.py --env env1 --show_sensing_range True --controller hexagon --original_method false
start cmd /c python scripts/run.py --env env1 --show_sensing_range True --controller hexagon --original_method true