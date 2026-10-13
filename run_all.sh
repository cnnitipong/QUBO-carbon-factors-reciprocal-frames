#!/bin/sh
set -e
python3 src/run_jobs.py 100000
python3 src/aggregate.py
python3 src/make_figures.py
python3 src/make_figures2.py
python3 src/make_figures3.py
