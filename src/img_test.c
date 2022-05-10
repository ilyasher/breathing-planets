#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "noise.h"
#include "libattopng.h"

#define RGBA(r, g, b, a) ((r) | ((g) << 8) | ((b) << 16) | ((a) << 24))
#define RGB(r, g, b) RGBA(r, g, b, 0xff)

#define ALPHA(c, a) ((c) | ((a) << 8))

#define W 450
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
    libattopng_t *png = libattopng_new(W, H, PNG_RGB);

    double r = 0.5;

    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {

            double x = (j - W/2)/(double)H;
            double y = (i - H/2)/(double)H;
            double xxyy = x*x+y*y;
            if (xxyy >= r*r) {
                libattopng_set_pixel(png, j, i, RGB(0, 0, 0));
                continue;
            }
            double z = sqrt(r*r-xxyy);
            double n1 = 4 * noise3(x*5, y*5, z*5);
            double n2 = 2 * noise3(x*10, y*10, z*10);
            double n3 = 1 * noise3(x*20, y*20, z*20);
            double n = (n1 + n2 + n3) / 7;
            if (n > -0.1) {
                libattopng_set_pixel(png, j, i, RGB(0, 0, 255));
            } else {
                libattopng_set_pixel(png, j, i, RGB(0, 200 - (int)(-(n - 0.3)*100), 0));
            }
        }
    }
    libattopng_save(png, "test_rgb.png");
    libattopng_destroy(png);
	return 0;
}