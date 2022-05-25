#!/bin/bash

time ./bin/render_planet earth_cpu.png -w 128 -h 128 --cpu
time ./bin/render_planet earth_cpu.gif -w 128 -h 128 --cpu --gif 10
time ./bin/render_planet earth_gpu.png -w 128 -h 128
time ./bin/render_planet earth_gpu.gif -w 128 -h 128 --gif 10
time ./bin/render_planet mars_cpu.png -w 128 -h 128 --cpu --mars
time ./bin/render_planet mars_cpu.gif -w 128 -h 128 --cpu --gif 10 --mars
time ./bin/render_planet mars_gpu.png -w 128 -h 128 --mars
time ./bin/render_planet mars_gpu.gif -w 128 -h 128 --gif 10 --mars