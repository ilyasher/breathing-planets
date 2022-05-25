#ifndef PLANET_GRAPHICS_CUH
#define PLANET_GRAPHICS_CUH

#include <stdlib.h>
#include <stdint.h>

void cuda_make_zs(float *zs, uint8_t *zs_valid, int W, int H, float t);

void cuda_fill_texture(void *pixels, float *zs, uint8_t *zs_valid, int W, int H, float t, int is_png);

void cuda_draw_planet(void *pixels, int W, int H, float t, int is_png);

#endif // PLANET_GRAPHICS_CUH
