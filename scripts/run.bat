@echo off

:: Original method
:: start python scripts/run.py --env env0 --show_sensing_range True --controller hexagon --original_method True

:: PSO method
:: start python scripts/run.py --env env0 --show_sensing_range True --controller hexagon

:: Voronoi method

:: start cmd /k python scripts/run.py --env env0 --show_connections true --show_sensing_range True --controller voronoi

:: start cmd /k python scripts/run.py --env env1 --show_connections true --show_sensing_range True --controller voronoi

:: start cmd /k python scripts/run.py --env env2 --show_connections true --show_sensing_range True --controller voronoi

:: start cmd /k python scripts/run.py --env env3 --show_connections true --show_sensing_range True --controller voronoi

:: start cmd /k python scripts/run.py --env env4 --show_connections true --show_sensing_range true --controller voronoi

::start cmd /k python scripts/run.py --env env0 --show_sensing_range True --controller voronoi

::start cmd /k python scripts/run.py --env env1 --show_sensing_range True --controller voronoi

::start cmd /k python scripts/run.py --env env2 --show_sensing_range True --controller voronoi

::start cmd /k python scripts/run.py --env env3 --show_sensing_range True --controller voronoi

start cmd /k python scripts/run.py --env env3 --show_sensing_range true --controller voronoi
start cmd /k python scripts/run.py --env env3 --show_sensing_range true --controller hexagon --original_method false

start cmd /k python scripts/run.py --env env4 --show_sensing_range true --controller voronoi
start cmd /k python scripts/run.py --env env4 --show_sensing_range true --controller hexagon --original_method false
