#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "noise.h"
#include "libattopng.h"
#include "gifenc.h"

#define RGBA(r, g, b, a) ((r) | ((g) << 8) | ((b) << 16) | ((a) << 24))
#define RGB(r, g, b) RGBA(r, g, b, 0xff)

#define ALPHA(c, a) ((c) | ((a) << 8))

#define W 400
#define H 400

int main(int argc, char *argv[])
{
    // srand(time(0));
    // double r = 3;
    // for (int i = -20; i < 20; i++) {
    //     for (int j = -40; j < 40; j++) {
    //         double x = i / 5.;
    //         double y = j / 10.;
    //         double xxyy = x*x+y*y;
    //         if (xxyy >= r*r) {
    //             printf(" ");
    //             continue;
    //         }
    //         double z = sqrt(r*r-xxyy);
    //         double n = noise3(x, y, z);
    //         if (n > 0) {
    //             printf("#");
    //         } else {
    //             printf(".");
    //         }
    //     }
    //     printf("\n");
    // }


    uint8_t palette[] = {
        0x00, 0x00, 0x00,   /* entry 0: black */
        170, 44, 23,   /* entry 1: dark orange */
        234, 136, 52,   /* entry 2: orange */
        255, 255, 255,   /* entry 3: white */
    };
    int depth = 2;

    ge_GIF *gif = ge_new_gif(
        "test_rgb.gif", W, H,
       palette, depth, -1, 1
    );

    double r = 0.5;

    uint8_t *pixels = malloc(W*H*sizeof(uint8_t));
    memset(pixels, 0, W*H*sizeof(uint8_t));

    for (int f = 0; f < 20; f++) {
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {

                double x = (j - W/2)/(double)H;
                double y = (i - H/2)/(double)H;
                double xxyy = x*x+y*y;
                if (xxyy >= r*r) {
                    continue;
                }
                double z = sqrt(r*r-xxyy);
                z += 0.01 * f;
                double n = 0;
                int power = 1;
                for (int freq = 10; freq < 80; freq *= 2) {
                    n += noise3(x*freq, y*freq, z*freq);
                    power *= 2;
                }
                n /= (power - 1);
                uint8_t c = (uint8_t) ((n + 0.55) * 4);
                pixels[i*H+j] = c;
            }
        }
        memcpy(gif->frame, pixels, W*H*sizeof(uint8_t));
        ge_add_frame(gif, 10);
    }
    ge_close_gif(gif);

	return 0;
}