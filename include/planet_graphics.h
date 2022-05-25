#ifndef PLANET_GRAPHICS_H
#define PLANET_GRAPHICS_H

#include <stdlib.h>
#include <stdint.h>

#define R 0.45
#define HEIGHT_AMP 0.1
#define ROTATION_FREQ -0.7

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

// extern float water_level;
// extern float (*height_function)(float);
// extern int   (*color_function)(float, float, float, float, int);
// extern float seed;


// Do the ray marching for every x, y and fill z values in zs
void make_zs(float *zs, uint8_t *zs_valid, int W, int H, float t, int planet);

//
void fill_texture(void *pixels, // either png object or gif frame
                  float *zs,
                  uint8_t *zs_valid,
                  int W,
                  int H,
                  float t,
                  int is_png,
                  int planet);

#endif // PLANET_GRAPHICS_H
