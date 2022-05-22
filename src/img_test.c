#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "noise.h"
#include "libattopng.h"
#include "gifenc.h"
#include "palette.h"

#define RGBA(r, g, b, a) ((r) | ((g) << 8) | ((b) << 16) | ((a) << 24))
#define RGB(r, g, b) RGBA(r, g, b, 0xff)

#define ALPHA(c, a) ((c) | ((a) << 8))

#define W 256
#define H 256

double r = 0.45;
double noise_amp = 0.1;

// Simplex noise added up at a few octaves
double noise(double x, double y, double z, double t) {
    double power = 1;
    double n = 0;
    for (int freq = 4; freq < 128; freq *= 2) {
        n += power * noise4(x*freq, y*freq, z*freq, t);
        power /= 2;
    }
    n *= 0.5 * sqrt(fabs(n)); // makes the shape a little bit spikier
    return n;
}


// Note: we might remove
double noise_grad_z(double x, double y, double z, double t) {
    double delta = 1e-6;
    double p1 = noise(x, y, z-delta, t);
    double p2 = noise(x, y, z+delta, t);
    return (p2 - p1) / (2 * delta);
}

// the function we want to find the zero of.
// If f(x, y, z, t) = 0, then (x, y, z) is at the planet's surface at time t
double f(double x, double y, double z, double t) {
    return x*x+y*y+z*z-(pow(r+noise_amp*noise(x,y,z,t), 2));
}

// Numerical derivatives
double f_grad_x(double x, double y, double z, double t) {
    double delta = 1e-6;
    double p1 = f(x-delta, y, z, t);
    double p2 = f(x+delta, y, z, t);
    return (p2 - p1) / (2 * delta);
}

double f_grad_y(double x, double y, double z, double t) {
    double delta = 1e-6;
    double p1 = f(x, y-delta, z, t);
    double p2 = f(x, y+delta, z, t);
    return (p2 - p1) / (2 * delta);
}

double f_grad_z(double x, double y, double z, double t) {
    double delta = 1e-6;
    double p1 = f(x, y, z - delta, t);
    double p2 = f(x, y, z + delta, t);
    return (p2 - p1) / (2 * delta);
}


// Note: f_grad_z does essentially the same thing
double f_prime(double x, double y, double z, double t) {
    return 2*z-2*(r+noise_amp*noise(x, y, z, t))*noise_amp*noise_grad_z(x, y, z, t);
}

// Given x, y, find which z is at the planet's surface
// (try find z such that f(x, y, z, t) = 0)
double find_z(double x, double y, double z, double t) {
    // double highest_z = -0.5;
    double highest_z = 1000;
    for (double offset = 0.2; offset >= -0.2; offset -= 0.05) {
        double new_z = z + offset;
        for (int i = 0; i < 8; i++) {
            new_z = new_z -  f(x, y, new_z, t) / (f_prime(x, y, new_z, t) + 1e-3);
        }

        if (f(x, y, new_z, t) < 1e-3) {
            highest_z = new_z;
            break;
        }
    }

    return highest_z;
}

// Faster but worse
// double find_z(double x, double y, double z) {
//     double new_z = z + 0.1;
//     for (int i = 0; i < 8; i++) {
//         new_z = new_z -  f(x, y, new_z) / (f_prime(x, y, new_z) + 1e-3);
//     }
//     return new_z;
// }

// Do the raytracing for every x, y and fill z values in zs
void make_zs(double zs[H][W], uint8_t zs_valid[H][W], double t) {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            double x = (j - W/2)/(double)H;
            double y = (i - H/2)/(double)H;
            double rrxxyy = r*r-x*x+y*y;
            double z = 0;
            if (rrxxyy > 0) {
                z = sqrt(rrxxyy);
            }
            z = find_z(x, y, z, t);
            zs[i][j] = z;
            zs_valid[i][j] = (fabs(f(x, y, z, t)) < 1e-3);
        }
    }
}

// For invalid locations that have valid neighbors,
// replace them with the average of the valid neighbors.
void fill_holes(double zs[H][W], uint8_t v[H][W]) {

    uint8_t new_valid[H][W];
    memcpy(new_valid, v, H*W);

    for (int i = 1; i < H-1; i++) {
        for (int j = 1; j < W-1; j++) {
            if (v[i][j]) continue;
            int sum = v[i-1][j-1] + v[i-1][j] + v[i-1][j+1]
                    + v[i][j-1]   +           + v[i][j+1]
                    + v[i+1][j-1] + v[i+1][j] + v[i+1][j+1];
            if (sum > 2) {
                double dot = v[i-1][j-1] * zs[i-1][j-1]
                           + v[i-1][j]   * zs[i-1][j]
                           + v[i-1][j+1] * zs[i-1][j+1]
                           + v[i][j-1]   * zs[i][j-1]
                           + v[i][j]     * zs[i][j]
                           + v[i][j+1]   * zs[i][j+1]
                           + v[i+1][j-1] * zs[i+1][j-1]
                           + v[i+1][j]   * zs[i+1][j]
                           + v[i+1][j+1] * zs[i+1][j+1];
                zs[i][j] = dot / sum;
                new_valid[i][j] = 1;
            }
        }
    }
    memcpy(v, new_valid, H*W);
}


// Return number 0-1 representing how much lighting the point has.
double lighting(double x, double y, double z, double t, int flat) {
    // double lx = sin(t);
    // double lz = cos(t);
    double lx = -0.7;
    double ly = 0.2;
    double lz = 0.7;

    double normx, normy, normz;
    if (flat) {
        normx = x/r;
        normy = y/r;
        normz = z/r;
    } else {
        double gradx = f_grad_x(x, y, z, t);
        double grady = f_grad_y(x, y, z, t);
        double gradz = f_grad_z(x, y, z, t);
        normx = sin(atan(gradx));
        normy = sin(atan(grady));
        normz = sin(atan(gradz));
    }
    // double gradz = noise_grad_z(x, y, z, t);

    double l = lx*normx + ly*normy + lz*normz;
    l = l > 0 ? l : 0;

    return 0.1 + 0.9 * l;
}

void fill_texture_png(libattopng_t *png, double zs[H][W], uint8_t zs_valid[H][W], double t) {
    double water_level = r - 0.01;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            if (!zs_valid[i][j]) {
                libattopng_set_pixel(png, j, i, RGB(0, 0, 0));
                continue;
            }

            double x = (j - W/2)/(double)H;
            double y = (i - H/2)/(double)H;
            double z = zs[i][j];


            double height = sqrt(x*x+y*y+z*z);
            double n = noise(x, y, z, t);

            double light = lighting(x, y, z, t, height < water_level);
            uint8_t c1 = (150+n*200) * light;

            if (height < water_level) {
                libattopng_set_pixel(png, j, i, RGB(c1, c1, (int) (255*light)));
                continue;
            }

            // double n = z;
            // double n = f(x, y, z);
            uint8_t c =  (int) (light * (150 + (int)((n)*200)));

            // Add other colors too here if you wish!
            libattopng_set_pixel(png, j, i, RGB(0, c, 0));
        }
    }
}

void fill_texture_gif(uint8_t *pixels, double zs[H][W], uint8_t zs_valid[H][W], double t) {
    double water_level = r - 0.01;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            if (!zs_valid[i][j]) {
                pixels[i*H+j] = BLACK;
                continue;
            }
            double x = (j - W/2)/(double)H;
            double y = (i - H/2)/(double)H;
            double z = zs[i][j];

            double height = sqrt(x*x+y*y+z*z);
            double n = noise(x, y, z, t);
            // uint8_t c1 =  100+n*200;

            double light = lighting(x, y, z, t, height < water_level);

            if (height < water_level) {
                int c = 48 + (int)(n*64);
                c = c > 64 ? 64 : (c < 0 ? 0 : c);
                pixels[i*H+j] = BLUE_TABLE[(int)(c*light)];
                continue;
            }

            // double n = z;
            // double n = f(x, y, z);
            // uint8_t c =  128 + (int)((n)*200);

            // Add other colors too here if you wish!
            int c = 16+(int)(n*96);
            c = c > 64 ? 64 : (c < 0 ? 0 : c);
            pixels[i*H+j] = GREEN_TABLE[(int)(c*light)];
        }
    }
}

int main(int argc, char *argv[])
{
    int do_gif = 1;

    // Array of z positions for every x, y
    double zs[H][W];

    // Array of whether z position is valid for every x, y
    // Invalid z positions occur either when the x, y corresponds to
    // outer space, or when the raytracing messes up.
    uint8_t zs_valid[H][W];


    if (!do_gif) {
        libattopng_t *png = libattopng_new(W, H, PNG_RGB);

        double t = 1; // arbitrary
        make_zs(zs, zs_valid, t);
        fill_holes(zs, zs_valid);
        fill_texture_png(png, zs, zs_valid, t);
        libattopng_save(png, "test_rgb_heights.png");
        libattopng_destroy(png);
        return 0;
    }
    else {

        ge_GIF *gif = ge_new_gif(
            "test_rgb.gif", W, H,
            palette, 8, -1, 1);

        // pixels contains the palette index of each x, y of the frame.
        uint8_t *pixels = malloc(W*H*sizeof(uint8_t));
        memset(pixels, 0, W*H*sizeof(uint8_t));

        int n_frames = 20;
        for (int f = 0; f < n_frames; f++) {
            printf("%d/%d\n", f, n_frames);
            double t = f / 40.;

            make_zs(zs, zs_valid, t);
            fill_holes(zs, zs_valid);
            fill_texture_gif(pixels, zs, zs_valid, t);

            // Necessary to do for every frame separately
            memcpy(gif->frame, pixels, W*H*sizeof(uint8_t));
            ge_add_frame(gif, 10);
        }
        ge_close_gif(gif);
        free(pixels);
    }

	return 0;
}