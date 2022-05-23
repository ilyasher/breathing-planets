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

#define CAPAT(c, min, max) ((c) < (min) ? (min) : ((c) > (max) ? (max) : (c)))

#define W 256
#define H 256

typedef struct vec3_t {
    float x;
    float y;
    float z;
} vec3_t;

float earth_height_function(float n);
float mars_height_function(float n);
int mars_color_function(float x, float y, float z, float t, int is_png);
int earth_color_function(float x, float y, float z, float t, int is_png);

float r = 0.45;
float noise_amp = 0.1;
float water_level = -1;
// float water_level = -0.005;
float snow_level = 0.025;

float rotation_freq = -0.7;

float (*height_function)(float) = mars_height_function;
int   (*color_function)(float, float, float, float, int) = mars_color_function;

// Simplex noise added up at a few octaves
float noise(float x, float y, float z, float t) {

    float x_rot = x*cos(rotation_freq * t) - z*sin(rotation_freq * t);
    float z_rot = x*sin(rotation_freq * t) + z*cos(rotation_freq * t);
    float y_rot = y;

    float power = 1;
    float n = 0;
    for (int freq = 2; freq < 256; freq *= 2) {
        n += power * noise4(x_rot*freq, y_rot*freq, z_rot*freq, 0);
        power /= 2.2;
    }

    n = height_function(n);
    return n;
}

float surface_height(float x, float y, float z, float t) {
    float norm = sqrt(x*x+y*y+z*z);
    float height = noise_amp*noise(r*x/norm, r*y/norm, r*z/norm, t);
    return height;
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
    for (int i = 0; i < 32; i++) {
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
            z = ray_march_z(x, y, z, t);
            zs[i][j] = z;
            zs_valid[i][j] = (fabs(sdf(x, y, z, t)) < 1e-2);
        }
    }
}

vec3_t lp = {-7, 2, 7};

// Return number 0-1 representing how much lighting the point has.
float lighting(float x, float y, float z, float t) {

    float ambient = 0.1;

    vec3_t ray = {lp.x * cos(t), lp.y, lp.z -14*sin(t)};
    vec3_t ld = {x-ray.x, y-ray.y, z-ray.z};
    normalize(&ld);
    for (int i = 0; i < 32; i++) {
        float dist = sdf(ray.x, ray.y, ray.z, t);
        ray.x += dist*ld.x;
        ray.y += dist*ld.y;
        ray.z += dist*ld.z;
    }
    float dist = sqrt((ray.x-x)*(ray.x-x)+(ray.y-y)*(ray.y-y)+(ray.z-z)*(ray.z-z));
    if (dist > 1e-2) {
        // In the shadow
        return ambient;
    }

    vec3_t grad;
    grad_sdf(&grad, x, y, z, t);

    float diffuse = -ld.x*grad.x - ld.y*grad.y - ld.z*grad.z;
    diffuse = diffuse > 0 ? diffuse : 0;

    // float specular = grad.z > 0 ? grad.z : 0;
    // specular = pow(specular, 256);

    return ambient + 0.9 * diffuse;
}

void fill_texture(void *pixels, float zs[H][W], uint8_t zs_valid[H][W], float t, int is_png) {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            if (!zs_valid[i][j]) {
                if (is_png) {
                    libattopng_set_pixel((libattopng_t *) pixels, j, i, RGB(0, 0, 0));
                } else {
                    ((uint8_t *)pixels)[i*H+j] = BLACK;
                }
                continue;
            }
            float x = (j - W/2)/(float)H;
            float y = (i - H/2)/(float)H;
            float z = zs[i][j];
            int color = color_function(x, y, z, t, is_png);
            if (is_png) {
                libattopng_set_pixel((libattopng_t *) pixels, j, i, color);
            } else {
                ((uint8_t *)pixels)[i*H+j] = color;
            }
        }
    }
}

float earth_height_function(float n) {
    return n * sqrt(fabs(n)); // makes the shape a little bit spikier
}

float mars_height_function(float n) {
    n += 0.3;
    n = -1./(500*n*n+4);// + 1/(1000*(x-0.3)*(x-0.3)+5);
    return n * 0.5;
}

int mars_color_function(float x, float y, float z, float t, int is_png) {
    float light = lighting(x, y, z, t);
    float n = surface_height(x, y, z, t);
    if (is_png) {
        float intensity = (800+n*10000) * light;
        if (intensity < 500) {
            int c = CAPAT(intensity / 500. * 255., 0, 255);
            return RGB(c, 0, 0);
        } else {
            int c = CAPAT((intensity-500) / 500. * 255., 0, 255);
            return RGB(255, c, c);
        }
    }
    // else is gif
    return RED_TABLE[CAPAT((int)(light * (50 + 64*20*n)), 0, 63)];
}

int earth_color_function(float x, float y, float z, float t, int is_png) {
    float light = lighting(x, y, z, t);
    float n = surface_height(x, y, z, t);

    if (n < water_level) {
        if (is_png) {
            int c = CAPAT((48+n*64) * light, 0, 255);
            return RGB(c, c,  (int) (255*light));
        }
        int c = 48 + (int)(n*64);
        return BLUE_TABLE[CAPAT((int)(c*light), 0, 63)];
    }
    if (n < snow_level) {
        if (is_png) {
            int c = CAPAT((light * (100 + (int)(n*3000))), 0, 255);
            return RGB(0, c, 0);
        }
        int c = 16+(int)(n*96);
        return GREEN_TABLE[CAPAT((int)(c*light), 0, 63)];
    } else {
        if (is_png) {
            int c = CAPAT((light * (100 + (int)(n*3000))), 0, 255);
            return RGB(c, c, c);
        }
        int c = 63;
        return GREY_TABLE[(int)(c*light)];
    }
}

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