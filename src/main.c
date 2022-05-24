#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "libattopng.h"
#include "gifenc.h"

#include "palette.h"
#include "mars.h"
#include "earth.h"
#include "planet_graphics.h"
#include "planet_graphics.cuh"

void usage(char *progname) {
    printf("usage: %s out_file [--mars] [--earth] [-w <width>] [-h <height>] [--gif <n_frames>] [--seed <seed>]\n"
           "example: %s mars_movie.gif --mars -w 512 -h 512 --gif 64\n", progname, progname);
    exit(1);
}

int main(int argc, char *argv[])
{
    int width = 256;
    int height = 256;
    int planet = 0; // 0 = earth, 1 = mars
    char *filename;
    int filename_found = 0;
    int do_gif = 0;
    int n_frames = 10;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-w")) {
            i++;
            width = atoi(argv[i]);
        } else if (!strcmp(argv[i], "-h")) {
            i++;
            height = atoi(argv[i]);
        } else if (!strcmp(argv[i], "--mars")) {
            planet = 1;
        } else if (!strcmp(argv[i], "--earth")) {
            planet = 0;
        } else if (!strcmp(argv[i], "--gif")) {
            do_gif = 1;
            i++;
            n_frames = atoi(argv[i]);
        } else if (!strcmp(argv[i], "--seed")) {
            i++;
            seed = atoi(argv[i]); // declared in planet_graphics.h
        } else {
            if (!filename_found) {
                filename_found = 1;
                filename = argv[i];
            } else {
                usage(argv[0]);
            }
        }
    }
    if (!filename_found) usage(argv[0]);

    if (planet == 0) {
        height_function = earth_height_function;
        color_function = earth_color_function;
        water_level = EARTH_WATER_LEVEL;
    } else {
        height_function = mars_height_function;
        color_function = mars_color_function;
        water_level = MARS_WATER_LEVEL;
    }

    // Array of z positions for every x, y
    float *zs = malloc(sizeof(float) * width * height);

    // Array of whether z position is valid for every x, y
    // Invalid z positions occur either when the x, y corresponds to
    // outer space, or when the raytracing messes up.
    uint8_t *zs_valid = malloc(sizeof(uint8_t) * width * height);


    if (!do_gif) {
        libattopng_t *png = libattopng_new(width, height, PNG_RGB);

        float t = 0; // arbitrary
        cuda_make_zs(zs, zs_valid, width, height, t);
        // make_zs(zs, zs_valid, width, height, t);
        fill_texture(png, zs, zs_valid, width, height, t, 1);
        libattopng_save(png, filename);
        libattopng_destroy(png);
    }
    else {
        ge_GIF *gif = ge_new_gif(
            filename, width, height,
            palette, 8, -1, 1);

        // pixels contains the palette index of each x, y of the frame.
        uint8_t *pixels = malloc(width*height*sizeof(uint8_t));
        memset(pixels, 0, width*height*sizeof(uint8_t));

        for (int f = 0; f < n_frames; f++) {
            printf("%d/%d\n", f, n_frames);
            float t = f / 40.;
            make_zs(zs, zs_valid, width, height, t);
            fill_texture(pixels, zs, zs_valid, width, height, t, 0);

            // Necessary to do for every frame separately
            memcpy(gif->frame, pixels, width*height*sizeof(uint8_t));
            ge_add_frame(gif, 10);
        }
        ge_close_gif(gif);
        free(pixels);
    }
    free(zs);
    free(zs_valid);
	return 0;
}