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

