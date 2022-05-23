#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "noise.h"
#include "libattopng.h"
#include "gifenc.h"

#include "palette.h"
#include "mars.h"
#include "earth.h"
#include "planet_graphics.h"

int main(int argc, char *argv[])
{
    int do_gif = 1;

    // Array of z positions for every x, y
    float zs[H][W];

    // Array of whether z position is valid for every x, y
    // Invalid z positions occur either when the x, y corresponds to
    // outer space, or when the raytracing messes up.
    uint8_t zs_valid[H][W];


    if (!do_gif) {
        libattopng_t *png = libattopng_new(W, H, PNG_RGB);

        float t = 0; // arbitrary
        make_zs(zs, zs_valid, t);
        fill_texture(png, zs, zs_valid, t, 1);
        libattopng_save(png, "mars_test.png");
        libattopng_destroy(png);
        return 0;
    }
    else {

        ge_GIF *gif = ge_new_gif(
            "mars_night_rot.gif", W, H,
            palette, 8, -1, 1);

        // pixels contains the palette index of each x, y of the frame.
        uint8_t *pixels = malloc(W*H*sizeof(uint8_t));
        memset(pixels, 0, W*H*sizeof(uint8_t));

        // int n_frames = 40*6.28;
        int n_frames = 10;
        for (int f = 0; f < n_frames; f++) {
            printf("%d/%d\n", f, n_frames);
            float t = f / 40.;

            make_zs(zs, zs_valid, t);
            fill_texture(pixels, zs, zs_valid, t, 0);

            // Necessary to do for every frame separately
            memcpy(gif->frame, pixels, W*H*sizeof(uint8_t));
            ge_add_frame(gif, 10);
        }
        ge_close_gif(gif);
        free(pixels);
    }

	return 0;
}