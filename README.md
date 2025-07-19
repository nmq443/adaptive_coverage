# Distributed Coverage Control

This repo contains the code for my final year thesis.

## TODO

* Separate compute_voronoi, draw_voronoi into 1 module
* Only scale variables (meters -> pixels) when rendering, not when computing
* Finding new node on the traveling way

## Requirements

This work is developed in Python 3.10, with libraries specified in `requirements.txt`. To install, just run
`pip install -r requirements.txt`. After that, run `pip install -e .` to install this project as a package.

## How to run

For now, just run `cd scripts && python3 main.py`
