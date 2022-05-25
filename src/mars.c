#include "mars.h"
#include "planet_graphics.h"
#include "palette.h"

float mars_height_function(float n) {
    n += 0.3;
    n = -1./(500*n*n+4);// + 1/(1000*(x-0.3)*(x-0.3)+5);
    return n * 0.5;
}

int mars_color_function(float x, float y, float z, float t, int is_png) {
    float light = lighting(x, y, z, t, 1);
    float n = surface_height(x, y, z, t, 1);
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