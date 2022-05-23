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

float HEIGHT_AMP = 0.1;
float WATER_LEVEL = -1;
// float WATER_LEVEL = -0.005;

float rotation_freq = -0.7;

vec3_t lp = {-7, 2, 7};

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
    float height = HEIGHT_AMP*noise(R*x/norm, R*y/norm, R*z/norm, t);
    return height;
}

float sdf(float x, float y, float z, float t) {
    float norm = sqrt(x*x+y*y+z*z);
    float h = surface_height(x, y, z, t);
    return norm - (R + (h > WATER_LEVEL ? h : WATER_LEVEL));
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
            float rrxxyy = R*R-x*x+y*y;
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