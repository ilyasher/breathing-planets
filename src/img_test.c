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

#define W 200
#define H 200

double r = 0.5;
double noise_amp = 0.1;

double noise(double x, double y, double z) {
    double power = 1;
    double n = 0;
    for (int freq = 4; freq < 64; freq *= 2) {
        n += power * noise3(x*freq, y*freq, z*freq);
        power /= 2;
    }
    // n /= (power - 1);

    // n += 0.1;
    n *= 0.5 * sqrt(fabs(n));
    return n;
}

double noise_grad_z(double x, double y, double z) {
    double delta = 1e-6;
    double p1 = noise(x, y, z-delta);
    double p2 = noise(x, y, z+delta);
    return (p2 - p1) / (2 * delta);
}

double f(double x, double y, double z) {
    return x*x+y*y+z*z-(pow(r+noise_amp*noise(x,y,z), 2));
}

double f_prime(double x, double y, double z) {
    return 2*z-2*(r+noise_amp*noise(x, y, z))*noise_amp*noise_grad_z(x, y, z);
}

double find_z(double x, double y, double z) {
    double higest_z = -0.5;
    for (double offset = -0.1; offset <= 0.1; offset += 0.01) {
        double new_z = z + offset;
        for (int i = 0; i < 8; i++) {

            new_z = new_z -  f(x, y, new_z) / (f_prime(x, y, new_z) + 1e-3);
        }

        if (f(x, y, new_z) < 1e-3) {
            higest_z = new_z;
        }
    }


    return higest_z;
}

void make_zs(double zs[H][W], uint8_t zs_valid[H][W]) {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            double x = (j - W/2)/(double)H;
            double y = (i - H/2)/(double)H;
            double rrxxyy = r*r-x*x+y*y;
            double z = 0;
            if (rrxxyy > 0) {
                z = sqrt(rrxxyy);
            }
            z = find_z(x, y, z);
            zs[i, j] = z;
            zs_valid[i, j] = (fabs(f(x, y, z)) < 1e-3);
            // double n = noise(x, y, z);
            // if (n < -10.1) {
                // libattopng_set_pixel(png, j, i, RGB(0, 0, 255));
            // } else {
            // uint8_t c = 250 - (int)(-(n - 0.3)*400);
            // uint8_t c =  128 + (int)((n)*200);

            // if (fabs(f(x, y, z)) < 1e-3) {

            //     libattopng_set_pixel(png, j, i, RGB(0, c, 0));
            //     // libattopng_set_pixel(png, j, i, RGB(0, (int) (100000 * f(x, y, z)), 0));
            // } else {
            //     libattopng_set_pixel(png, j, i, RGB(0, 0, 0));
            // }
            // }
        }
    }
}

void fill_holes(double zs[H][W], uint8_t v[H][W]) {

    for (int i = 1; i < H-1; i++) {
        for (int j = 1; j < W-1; j++) {
            if (v[i][j]) continue;
            int sum = v[i-1][j-1] + v[i-1][j] + v[i-1][j+1]
                    + v[i][j-1]   +           + v[i][j+1]
                    + v[i+1][j-1] + v[i+1][j] + v[i+1][j+1];
            if (sum > 0) {
                double dot = v[i-1][j-1] * zs[i-1][j-1]
                           + v[i-1][j]   * zs[i-1][j]
                           + v[i-1][j+1] * zs[i-1][j-1]
                           + v[i][j-1]   * zs[i][j-1]
                           + v[i][j]     * zs[i][j]
                           + v[i][j+1]   * zs[i][j-1]
                           = v[i+1][j-1] * zs[i+1][j-1]
                           + v[i+1][j]   * zs[i+1][j]
                           + v[i+1][j+1] * zs[i+1][j-1];
                zs[i][j] = dot / sum;
            }
        }
    }
}


int main(int argc, char *argv[])
{


    // uint8_t palette[] = {
    //     0x00, 0x00, 0x00,   /* entry 0: black */
    //     170, 44, 23,   /* entry 1: dark orange */
    //     234, 136, 52,   /* entry 2: orange */
    //     255, 255, 255,   /* entry 3: white */
    // };
    // int depth = 2;

    // ge_GIF *gif = ge_new_gif(
    //     "test_rgb.gif", W, H,
    //    palette, depth, -1, 1
    // );

    libattopng_t *png = libattopng_new(W, H, PNG_RGB);

    double zs[H, W] = {0.};
    uint8_t zs_valid[H, W] = {false};

    make_zs(zs, zs_valid);
    fill_holes(zs, zs_valid);
    fill_texture(png, zs, zs_valid);

    // libattopng_save(png, "test_rgb.png");
    libattopng_save(png, "test_rgb_heights.png");
    libattopng_destroy(png);
	return 0;

/*
    ge_GIF *gif = ge_new_gif(
        "test_rgb.gif", W, H,
       NULL, 8, -1, 1
    );

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
                x += 0.01 * f;
                z = find_z(x, y, z);
                double n = noise(x, y, z);

                uint8_t c = (uint8_t) ((n + 0.55) * 50);
                pixels[i*H+j] = c;
            }
        }
        memcpy(gif->frame, pixels, W*H*sizeof(uint8_t));
        ge_add_frame(gif, 10);
    }
    ge_close_gif(gif);

	return 0;
    */
}