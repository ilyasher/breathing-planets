#ifndef PLANET_GRAPHICS_H
#define PLANET_GRAPHICS_H

#include <stdlib.h>

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
void normalize(vec3_t* v);

extern float water_level;
extern float (*height_function)(float);
extern int   (*color_function)(float, float, float, float, int);

// Functions relating to planet generation

// Simplex noise added up at a few octaves
float noise(float x, float y, float z, float t);

// Project (x,y,z) down onto the planet and see what the height of the surface is there.
float surface_height(float x, float y, float z, float t);

// Signed distance function. Distance to the planet's surface.
float sdf(float x, float y, float z, float t);

// Return the numerical gradient of sdf at (x, y, z) in 'grad'
void grad_sdf(vec3_t *grad, float x, float y, float z, float t);

// Given x, y, find which z is at the planet's surface
// (try find z such that sgd(x, y, z, t) = 0)
float ray_march_z(float x, float y, float z, float t);


// Return number 0-1 representing how much lighting the point has.
float lighting(float x, float y, float z, float t);


// Do the ray marching for every x, y and fill z values in zs
void make_zs(float *zs, uint8_t *zs_valid, int W, int H, float t);

//
void fill_texture(void *pixels, // either png object or gif frame
                  float *zs,
                  uint8_t *zs_valid,
                  int W,
                  int H,
                  float t,
                  int is_png);

#endif // PLANET_GRAPHICS_H
