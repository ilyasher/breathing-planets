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
    printf("usage: %s out_file [--mars] [--earth] [-w <width>] [-h <height>] [--gif <n_frames>] [--seed <seed>] [--cpu]\n"
           "example: %s mars_movie.gif --mars -w 512 -h 512 --gif 64\n", progname, progname);
    exit(1);
}

float get_offset_from_seed(int seed) {
    srand(seed);
    return 1000 * (rand() / (float) RAND_MAX);
}

int main(int argc, char *argv[])
{
    int width = 256;
    int height = 256;
    int planet = 0; // 0 = earth, 1 = mars
    char *filename;
    int filename_found = 0;
    int do_png = 1;
    int do_gpu = 1;
    int n_frames = 10;
    int seed = 0;

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
            do_png = 0;
            i++;
            n_frames = atoi(argv[i]);
        } else if (!strcmp(argv[i], "--seed")) {
            i++;
            seed = atoi(argv[i]); // declared in planet_graphics.h
        } else if (!strcmp(argv[i], "--cpu")) {
            do_gpu = 0;
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

    float offset = get_offset_from_seed(seed);

    // Array of z positions for every x, y
    float *zs;

    // Array of whether z position is valid for every x, y
    // Invalid z positions occur either when the x, y corresponds to
    // outer space, or when the raytracing messes up.
    uint8_t *zs_valid;

    if (!do_gpu) {
        zs = malloc(sizeof(float) * width * height);
        zs_valid = malloc(sizeof(uint8_t) * width * height);
    }

    if (do_png) {
        libattopng_t *png = libattopng_new(width, height, PNG_RGB);

        float t = 0; // arbitrary
        if (do_gpu) {
            cuda_draw_planet(png->data, width, height, t, do_png, planet, offset);
        } else {
            make_zs(zs, zs_valid, width, height, t, planet, offset);
            fill_texture(png, zs, zs_valid, width, height, t, do_png, planet, offset);
        }
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

            if (do_gpu) {
                cuda_draw_planet(pixels, width, height, t, do_png, planet, offset);
            } else {
                make_zs(zs, zs_valid, width, height, t, planet, offset);
                fill_texture(pixels, zs, zs_valid, width, height, t, do_png, planet, offset);
            }

            // Necessary to do for every frame separately
            memcpy(gif->frame, pixels, width*height*sizeof(uint8_t));
            ge_add_frame(gif, 10);
        }
        ge_close_gif(gif);
        free(pixels);
    }

    if (!do_gpu) {
        free(zs);
        free(zs_valid);
    }

	return 0;
}