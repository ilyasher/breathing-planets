#ifndef PLANET_GRAPHICS_CUH
#define PLANET_GRAPHICS_CUH

#include <stdlib.h>
#include <stdint.h>

void cuda_draw_planet(void *pixels, int W, int H, float t, int is_png, int planet, float offset);

#endif // PLANET_GRAPHICS_CUH
