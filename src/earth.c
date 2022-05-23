#include <math.h>

#include "earth.h"
#include "planet_graphics.h"
#include "noise.h"
#include "palette.h"

float earth_height_function(float n) {
    return n * sqrt(fabs(n)); // makes the shape a little bit spikier
}

int earth_color_function(float x, float y, float z, float t, int is_png) {
    float light = lighting(x, y, z, t);
    float n = surface_height(x, y, z, t);

    if (n < EARTH_WATER_LEVEL) {
        if (is_png) {
            int c = CAPAT((48+n*64) * light, 0, 255);
            return RGB(c, c,  (int) (255*light));
        }
        int c = 48 + (int)(n*64);
        return BLUE_TABLE[CAPAT((int)(c*light), 0, 63)];
    }
    if (n + fabs(y / 10) + 0.01 * noise4(256*x,256*y,256*z,t) < EARTH_SNOW_LEVEL) {
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