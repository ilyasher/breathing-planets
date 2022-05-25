# breathing-planets

A terrain generator and raytracer for making planet renders. The program can make GIFS of the planets changing over time.

Compile with

    $ make clean all

Program usage:

    $ ./bin/render_planet out_file_name [--mars] [-w <width>] [-h <height>] [--gif <n_frames>] [--seed <seed>] [--cpu]

`out_file_name` The name of the resulting image file. If `--gif` is specified, the name should end in `.gif`, otherwise it should end in `.png`.

`--mars` If specified, the rendered planet is Mars-like, otherwise it is Earth-like.

`-w <width> -h <height>` The width and height of the image, default for both is 256. Note that the program expects `width >= height`.

`--gif <n_frames>` If specified, the image created is an animated GIF with `n_frames` frames.

`--seed <seed>` Used to create different images from run to run. Default is 0.

`--cpu` If specified, code is executed only on the CPU, otherwise the GPU is used.

# Example usages

    $ ./bin/render_planet earth_large.gif -w 640 -h 512 --gif 451

    $ ./bin/render_planet mars_still.png -w 4096 -h 4096 --mars

# Example outputs

![Earth GIF](/imgs/example_earth.gif "Rendered spinning Earth")

![Mars PNG](/imgs/mars_large.png "Rendered Mars")

# GPU vs CPU

The program runs much faster on the GPU than on the CPU. Rendering a 4096x4096 PNG took 445 seconds when done on the CPU,
but only 2.1 seconds on the GPU, a 200x improvement. Rendering a 256x256 100-frame GIF took 212 seconds on the CPU, but only 2.2 seconds on the GPU, a 100x improvement.

# How it Works

The planet is rendered using a technique called ray-marching. The planet is described by a 4D function called a Signed Distance Function, or a SDF for short. The first 3 inputs are spatial `(x, y, z)` coordinates, and the fourth input is a time coordinate. As the name implies, the SDF gives the distance of a point from the surface of the planet. If `SDF(x, y, z, t) < 0`, then `(x, y, z)` is inside the planet at time `t`. If `SDF(x, y, z, t) = 0`, then `(x, y, z)` is on the planet's surface at time `t`. And if `SDF(x, y, z, t) > 0`, then `(x, y, z)` is outside of the planet at time `t`. Once the SDF is constructed, ray-marching is performed as follows: Point a ray towards the planet. Iteratively move it forward by `SDF(x, y, z, t)` where `(x, y, z)` is the ray's current position. If after a while `SDF(x, y, z, t)` is close to 0, then the ray hit the planet at point `(x, y, z)`, otherwise the ray misses the planet.

`SDF(x, y, z, t)` for the planets was constructed as follows. First, project `(x, y, z)` downwards onto the surface of a sphere of some radius `R` to get a new point `(x', y', z')`. If the planet is smooth, then the SDF is simply the distance to this projected point. We make the planet's surface rough by adding `Noise(x', y', z', t)` to this "smooth" SDF, where `Noise` is 4-dimensional Simplex noise summed at different frequencies and amplitudes. By increasing the time coordinate with every frame, the planet's surface evolves very gradually and naturally, thanks to the magic of Simplex noise.

Lighting is the standard ambient + diffuse, with additional raytracing to create shadows from mountains.

_________________

![Earth Terrain Closeup](/imgs/large_earth_closeup.png "Earth Terrain Closeup")

