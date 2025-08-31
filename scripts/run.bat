@echo off

:: Original method
:: start python scripts/run.py --env env0 --show_sensing_range True --controller hexagon --original_method True

:: PSO method
:: start python scripts/run.py --env env0 --show_sensing_range True --controller hexagon

:: Voronoi method
start python scripts/run.py --env env1 --show_sensing_range True --show_connections True --controller voronoi --total_time 10000 --num_agents 40