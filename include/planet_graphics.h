#ifndef PLANET_GRAPHICS_H
#define PLANET_GRAPHICS_H

#include <stdlib.h>

#define W 256
#define H 256

#define R 0.45

#define RGBA(r, g, b, a) ((r) | ((g) << 8) | ((b) << 16) | ((a) << 24))
#define RGB(r, g, b) RGBA(r, g, b, 0xff)
#define ALPHA(c, a) ((c) | ((a) << 8))

#define CAPAT(c, min, max) ((c) < (min) ? (min) : ((c) > (max) ? (max) : (c)))

typedef struct vec3_t {
    float x;
    float y;
    float z;
} vec3_t;

extern float (*height_function)(float);
extern int   (*color_function)(float, float, float, float, int);

// Simplex noise added up at a few octaves
float noise(float x, float y, float z, float t);

float surface_height(float x, float y, float z, float t);

float sdf(float x, float y, float z, float t);

void normalize(vec3_t* v);

void grad_sdf(vec3_t *grad, float x, float y, float z, float t);

// Given x, y, find which z is at the planet's surface
// (try find z such that f(x, y, z, t) = 0)
float ray_march_z(float x, float y, float z, float t);

// Return number 0-1 representing how much lighting the point has.
float lighting(float x, float y, float z, float t);

// Do the raytracing for every x, y and fill z values in zs
void make_zs(float zs[H][W], uint8_t zs_valid[H][W], float t);

void fill_texture(void *pixels, float zs[H][W], uint8_t zs_valid[H][W], float t, int is_png);

#endif // PLANET_GRAPHICS_H