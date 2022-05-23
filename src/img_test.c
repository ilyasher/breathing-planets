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

#define CAPAT(c, min, max) ((c) < (min) ? (min) : ((c) > (max) ? (max) : (c)))

#define ALPHA(c, a) ((c) | ((a) << 8))

#define W 128
#define H 128

typedef struct vec3_t {
    float x;
    float y;
    float z;
} vec3_t;

float r = 0.45;
float noise_amp = 0.1;
float water_level = -0.005;
float snow_level = 0.025;

// Simplex noise added up at a few octaves
float noise(float x, float y, float z, float t) {
    float power = 1;
    float n = 0;
    for (int freq = 2; freq < 256; freq *= 2) {
        n += power * noise4(x*freq, y*freq, z*freq, t);
        power /= 2;
    }
    n *= sqrt(fabs(n)); // makes the shape a little bit spikier
    // n *= 2 * fabs(n); // makes the shape a little bit spikier
    return n;
}

float surface_height(float x, float y, float z, float t) {
    float norm = sqrt(x*x+y*y+z*z);
    float height = noise_amp*noise(r*x/norm, r*y/norm, r*z/norm, t);
    return height;
    // return height > water_level ? height : water_level;
}

float sdf(float x, float y, float z, float t) {
    float norm = sqrt(x*x+y*y+z*z);
    float h = surface_height(x, y, z, t);
    return norm - (r + (h > water_level ? h : water_level));
}

void normalize(vec3_t* v) {
    float norm = sqrt(v->x*v->x + v->y*v->y + v->z*v->z);
    v->x /= norm;
    v->y /= norm;
    v->z /= norm;
}

void grad_sdf(vec3_t *grad, float x, float y, float z, float t) {
    float delta = 1e-3;
    float p0 = sdf(x, y, z, t);
    float px = sdf(x+delta, y, z, t);
    float py = sdf(x, y+delta, z, t);
    float pz = sdf(x, y, z+delta, t);
    grad->x = (px - p0) / delta;
    grad->y = (py - p0) / delta;
    grad->z = (pz - p0) / delta;
    normalize(grad);
}

// Given x, y, find which z is at the planet's surface
// (try find z such that f(x, y, z, t) = 0)
float ray_march_z(float x, float y, float z, float t) {
    for (int i = 0; i < 16; i++) {
        z = z - sdf(x, y, z, t);
    }
    return z;
}

// Do the raytracing for every x, y and fill z values in zs
void make_zs(float zs[H][W], uint8_t zs_valid[H][W], float t) {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            float x = (j - W/2)/(float)H;
            float y = (i - H/2)/(float)H;
            float rrxxyy = r*r-x*x+y*y;
            float z = 0;
            if (rrxxyy > 0) {
                z = sqrt(rrxxyy);
            }
            // z = find_z(x, y, z, t);
            z = ray_march_z(x, y, z, t);
            zs[i][j] = z;
            // zs_valid[i][j] = (fabs(f(x, y, z, t)) < 1e-2);
            zs_valid[i][j] = (fabs(sdf(x, y, z, t)) < 1e-2);

            // zs_valid[i][j] = 1;
        }
    }
}

// Return number 0-1 representing how much lighting the point has.
float lighting(float x, float y, float z, float t) {
    // float lx = sin(t);
    // float lz = cos(t);
    float lx = -0.7;
    float ly = 0.2;
    float lz = 0.7;

    vec3_t grad;
    grad_sdf(&grad, x, y, z, t);

    float l = lx*grad.x + ly*grad.y + lz*grad.z;
    l = l > 0 ? l : 0;

    return 0.1 + 0.9 * l;
}

void fill_texture_png(libattopng_t *png, float zs[H][W], uint8_t zs_valid[H][W], float t) {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            if (!zs_valid[i][j]) {
                libattopng_set_pixel(png, j, i, RGB(0, 0, 0));
                continue;
            }

            float x = (j - W/2)/(float)H;
            float y = (i - H/2)/(float)H;
            float z = zs[i][j];

            float n = surface_height(x, y, z, t);

            float light = lighting(x, y, z, t);
            // float light = 1;
            int c1 = CAPAT((100+n*2000) * light, 0, 255);

            if (n < water_level) {
                libattopng_set_pixel(png, j, i, RGB(c1, c1, (int) (255*light)));
                continue;
            }

            if (n < snow_level) {
                int c =  CAPAT((light * (100 + (int)(n*3000))), 0, 255);
                // Add other colors too here if you wish!
                libattopng_set_pixel(png, j, i, RGB(0, c, 0));

            } else {
                int c =  CAPAT((int) (light * (100 + (int)(n*3000))), 0, 255);
                // Add other colors too here if you wish!
                libattopng_set_pixel(png, j, i, RGB(c, c, c));
            }
        }
    }
}

void fill_texture_gif(uint8_t *pixels, float zs[H][W], uint8_t zs_valid[H][W], float t) {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            if (!zs_valid[i][j]) {
                pixels[i*H+j] = BLACK;
                continue;
            }
            float x = (j - W/2)/(float)H;
            float y = (i - H/2)/(float)H;
            float z = zs[i][j];

            float n = surface_height(x, y, z, t);
            // uint8_t c1 =  100+n*200;

            float light = lighting(x, y, z, t);

            if (n < water_level) {
                int c = 48 + (int)(n*64);
                c = c > 64 ? 64 : (c < 0 ? 0 : c);
                pixels[i*H+j] = BLUE_TABLE[(int)(c*light)];
                continue;
            }

            // Add other colors too here if you wish!
            if (n < snow_level) {
                int c = 16+(int)(n*96);
                c = c > 64 ? 64 : (c < 0 ? 0 : c);
                pixels[i*H+j] = GREEN_TABLE[(int)(c*light)];

            } else {
                // int c = 24+(int)(n*96);
                int c = 63;
                pixels[i*H+j] = GREY_TABLE[(int)(c*light)];
            }
        }
    }
}

int main(int argc, char *argv[])
{
    int do_gif = 0;

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
            float t = f / 40.;

            make_zs(zs, zs_valid, t);
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