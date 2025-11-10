@echo off

start cmd /k python scripts/run.py --env env1 --show_sensing_range True --controller hexagon --original_method false
start cmd /k python scripts/run.py --env env2 --show_sensing_range True --controller hexagon --original_method false
start cmd /k python scripts/run.py --env env3 --show_sensing_range True --controller hexagon --original_method false
start cmd /k python scripts/run.py --env env4 --show_sensing_range True --controller hexagon --original_method false
start cmd /k python scripts/run.py --env env5 --show_sensing_range True --controller hexagon --original_method false
start cmd /k python scripts/run.py --env env6 --show_sensing_range True --controller hexagon --original_method false