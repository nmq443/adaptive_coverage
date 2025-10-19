# Distributed Coverage Control

This repo contains the code for my final year thesis.

## Requirements

This work is developed in Python 3.10. To install, create a virtual environment and run `pip install -e .` 

## Result arrays
* `swarm_data.npy`: a `num_agents` x `num_timesteps` x 9 (`pos_x`, `pos_y`, `theta`, `goal_x`, `goal_y`, `dx`, `dy`, `speed`, `penalty_flag`) array represents the state of the swarm over time.

* `ld2s_data.npy`: lambda2 value over time.

* `travel_distances.npy`: travel distances over time.

## How to run

For now, just run `cd scripts && python3 main.py`

## TODO
* Fix bug: critical agents of current agent is empty
* Expand one-connected topology