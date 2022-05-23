#ifndef EARTH_H
#define EARTH_H

float earth_height_function(float n);
int earth_color_function(float x, float y, float z, float t, int is_png);

#define EARTH_SNOW_LEVEL 0.025
#define EARTH_WATER_LEVEL 0.005

#endif // EARTH_H