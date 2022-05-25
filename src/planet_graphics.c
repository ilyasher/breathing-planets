#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "noise.h"
#include "libattopng.h"
#include "gifenc.h"
#include "palette.h"

#include "mars.h"
#include "earth.h"
#include "planet_graphics.h"

vec3_t lp = {-7, 2, 7};

#define INDEX(i, j, W) ((i) * (W) + (j))

// Simplex noise added up at a few octaves
float noise(float x, float y, float z, float t, int planet) {

    float x_rot = x*cos(ROTATION_FREQ * t) - z*sin(ROTATION_FREQ * t);
    float z_rot = x*sin(ROTATION_FREQ * t) + z*cos(ROTATION_FREQ * t);
    float y_rot = y;

    float power = 1;
    float n = 0;
    for (int freq = 2; freq < 256; freq *= 2) {
        n += power * noise4(x_rot*freq, y_rot*freq, z_rot*freq, t);
        power /= 2.2;
    }

    if (planet == 0) {
        n = earth_height_function(n);
    } else {
        n = mars_height_function(n);
    }
    return n;
}

float surface_height(float x, float y, float z, float t, int planet) {
    float norm = sqrt(x*x+y*y+z*z);
    float height = HEIGHT_AMP*noise(R*x/norm, R*y/norm, R*z/norm, t, planet);
    return height;
}

float sdf(float x, float y, float z, float t, int planet) {
    float norm = sqrt(x*x+y*y+z*z);
    float h = surface_height(x, y, z, t, planet);
    int water_level = planet == 0 ? EARTH_WATER_LEVEL : MARS_WATER_LEVEL;
    return norm - (R + (h > water_level ? h : water_level));
}

void normalize(vec3_t* v) {
    float norm = sqrt(v->x*v->x + v->y*v->y + v->z*v->z);
    v->x /= norm;
    v->y /= norm;
    v->z /= norm;
}

void grad_sdf(vec3_t *grad, float x, float y, float z, float t, int planet) {
    float delta = 1e-3;
    float p0 = sdf(x, y, z, t, planet);
    float px = sdf(x+delta, y, z, t, planet);
    float py = sdf(x, y+delta, z, t, planet);
    float pz = sdf(x, y, z+delta, t, planet);
    grad->x = (px - p0) / delta;
    grad->y = (py - p0) / delta;
    grad->z = (pz - p0) / delta;
    normalize(grad);
}

// Given x, y, find which z is at the planet's surface
// (try find z such that f(x, y, z, t) = 0)
float ray_march_z(float x, float y, float z, float t, int planet) {
    for (int i = 0; i < 32; i++) {
        z = z - sdf(x, y, z, t, planet);
    }
    return z;
}

// Return number 0-1 representing how much lighting the point has.
float lighting(float x, float y, float z, float t, int planet) {

    float ambient = 0.1;

    vec3_t ray = {lp.x * cos(t), lp.y, lp.z -14*sin(t)};
    vec3_t ld = {x-ray.x, y-ray.y, z-ray.z};
    normalize(&ld);
    for (int i = 0; i < 32; i++) {
        float dist = sdf(ray.x, ray.y, ray.z, t, planet);
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
    grad_sdf(&grad, x, y, z, t, planet);

    float diffuse = -ld.x*grad.x - ld.y*grad.y - ld.z*grad.z;
    diffuse = diffuse > 0 ? diffuse : 0;

    // float specular = grad.z > 0 ? grad.z : 0;
    // specular = pow(specular, 256);

    return ambient + 0.9 * diffuse;
}

// Do the raytracing for every x, y and fill z values in zs
void make_zs(float *zs, uint8_t *zs_valid, int W, int H, float t, int planet, float offset) {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            float x = (j - W/2)/(float)H;
            float y = (i - H/2)/(float)H + offset;
            float rrxxyy = R*R-x*x+y*y;
            float z = 0;
            if (rrxxyy > 0) {
                z = sqrt(rrxxyy);
            }
            z = ray_march_z(x, y, z, t, planet);
            zs[INDEX(i, j, W)] = z;
            zs_valid[INDEX(i, j, W)] = (fabs(sdf(x, y, z, t, planet)) < 1e-2);
        }
    }
}

void fill_texture(void *pixels, float *zs, uint8_t *zs_valid, int W, int H, float t, int is_png, int planet, float offset) {
    int (*color_function)(float, float, float, float, int) = planet == 0 ? earth_color_function : mars_color_function;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            if (!zs_valid[INDEX(i, j, W)]) {
                if (is_png) {
                    libattopng_set_pixel((libattopng_t *) pixels, j, i, RGB(0, 0, 0));
                } else {
                    ((uint8_t *)pixels)[i*W+j] = BLACK;
                }
                continue;
            }
            float x = (j - W/2)/(float)H;
            float y = (i - H/2)/(float)H + offset;
            float z = zs[INDEX(i, j, W)];
            int color = color_function(x, y, z, t, is_png);
            if (is_png) {
                libattopng_set_pixel((libattopng_t *) pixels, j, i, color);
            } else {
                ((uint8_t *)pixels)[i*W+j] = color;
            }
        }
    }
}