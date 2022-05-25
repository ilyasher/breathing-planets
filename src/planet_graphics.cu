#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "helper_cuda.h"
#include "planet_graphics.h"
#include "planet_graphics.cuh"
#include "noise.h"
#include "noise.cuh"
#include "palette.cuh"

#define EARTH_WATER_LEVEL -0.005
#define MARS_WATER_LEVEL -1
#define EARTH_SNOW_LEVEL 0.05

__device__ static vec3_t lp = {-7, 2, 7};
__constant__ float gpu_t_offset[1];

#define INDEX(i, j, W) ((i) * (W) + (j))

// Simplex noise added up at a few octaves
__device__ static float gpu_noise(float x, float y, float z, float t, int planet) {

    float x_rot = x*cos(ROTATION_FREQ * t) - z*sin(ROTATION_FREQ * t);
    float z_rot = x*sin(ROTATION_FREQ * t) + z*cos(ROTATION_FREQ * t);
    float y_rot = y;

    float power = 1;
    float n = 0;
    for (int freq = 2; freq < 256; freq *= 2) {
        n += power * gpu_noise4(x_rot*freq, y_rot*freq, z_rot*freq, t+gpu_t_offset[0]);
        power /= 2.2;
    }

    if (planet == 0) {
        n = n * sqrt(fabs(n));
    } else {
        n += 0.3;
        n = -0.5/(500*n*n+4);
    }
    return n;
}

__device__ static float gpu_surface_height(float x, float y, float z, float t, int planet) {
    float norm = sqrt(x*x+y*y+z*z);
    float height = HEIGHT_AMP*gpu_noise(R*x/norm, R*y/norm, R*z/norm, t, planet);
    return height;
}

__device__ static float gpu_sdf(float x, float y, float z, float t, int planet) {
    float norm = sqrt(x*x+y*y+z*z);
    float h = gpu_surface_height(x, y, z, t, planet);
    float water_level = planet == 0 ? EARTH_WATER_LEVEL : MARS_WATER_LEVEL;
    return norm - (R + (h > water_level ? h : water_level));
}

__device__ static void gpu_normalize(vec3_t* v) {
    float norm = sqrt(v->x*v->x + v->y*v->y + v->z*v->z);
    v->x /= norm;
    v->y /= norm;
    v->z /= norm;
}

__device__ static void gpu_grad_sdf(vec3_t *grad, float x, float y, float z, float t, int planet) {
    float delta = 1e-3;
    float p0 = gpu_sdf(x, y, z, t, planet);
    float px = gpu_sdf(x+delta, y, z, t, planet);
    float py = gpu_sdf(x, y+delta, z, t, planet);
    float pz = gpu_sdf(x, y, z+delta, t, planet);
    grad->x = (px - p0) / delta;
    grad->y = (py - p0) / delta;
    grad->z = (pz - p0) / delta;
    gpu_normalize(grad);
}

// Given x, y, find which z is at the planet's surface
// (try find z such that f(x, y, z, t) = 0)
__device__ static float gpu_ray_march_z(float x, float y, float z, float t, int planet) {
    for (int i = 0; i < 32; i++) {
        z = z - gpu_sdf(x, y, z, t, planet);
    }
    return z;
}

#define AMBIENT 0.1

// Return number 0-1 representing how much lighting the point has.
__device__ static float gpu_lighting(float x, float y, float z, float t, int planet) {

    vec3_t ray = {lp.x * cos(t), lp.y, lp.z -14*sin(t)};
    vec3_t ld = {x-ray.x, y-ray.y, z-ray.z};
    gpu_normalize(&ld);
    for (int i = 0; i < 32; i++) {
        float dist = gpu_sdf(ray.x, ray.y, ray.z, t, planet);
        ray.x += dist*ld.x;
        ray.y += dist*ld.y;
        ray.z += dist*ld.z;
    }
    if ((ray.x-x)*(ray.x-x)+(ray.y-y)*(ray.y-y)+(ray.z-z)*(ray.z-z) > 1e-4) {
        // In the shadow
        return AMBIENT;
    }
    vec3_t grad;
    gpu_grad_sdf(&grad, x, y, z, t, planet);

    float diffuse = -ld.x*grad.x - ld.y*grad.y - ld.z*grad.z;
    diffuse = diffuse > 0 ? diffuse : 0;

    return AMBIENT + 0.9 * diffuse;
}

__device__ static int gpu_earth_color_function(float x, float y, float z, float t, int is_png) {
    float light = gpu_lighting(x, y, z, t, 0);

    // float light = 0.5;
    float n = gpu_surface_height(x, y, z, t, 0);

    if (n < EARTH_WATER_LEVEL) {
        if (is_png) {
            int c = CAPAT((48+n*64) * light, 0, 255);
            return RGB(c, c,  (int) (255*light));
        }
        int c = 48 + (int)(n*64);
        return BLUE_TABLE_GPU[CAPAT((int)(c*light), 0, 63)];
    }
    if (n + fabs(y / 10) + 0.01 * gpu_noise4(256*x,256*y,256*z,t) < EARTH_SNOW_LEVEL) {
        if (is_png) {
            int c = CAPAT((light * (100 + (int)(n*3000))), 0, 255);
            return RGB(0, c, 0);
        }
        int c = 16+(int)(n*96);
        return GREEN_TABLE_GPU[CAPAT((int)(c*light), 0, 63)];
    } else {
        if (is_png) {
            int c = CAPAT((light * (100 + (int)(n*3000))), 0, 255);
            return RGB(c, c, c);
        }
        int c = 63;
        return GREY_TABLE_GPU[(int)(c*light)];
    }
    // return RGB(255, 0, 0);
}

__device__ static int gpu_mars_color_function(float x, float y, float z, float t, int is_png) {
    float light = gpu_lighting(x, y, z, t, 1);
    float n = gpu_surface_height(x, y, z, t, 1);
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
    return RED_TABLE_GPU[CAPAT((int)(light * (50 + 64*20*n)), 0, 63)];
}

__global__ void make_zs_kernel(float *d_zs, uint8_t *d_zs_valid, int W, int H, float t, int planet) {
    int i = threadIdx.x + 32 * blockIdx.x;
    int j = threadIdx.y + 16 * blockIdx.y;

    float x = (j - W/2)/(float)H;
    float y = (i - H/2)/(float)H;
    float rrxxyy = R*R-x*x+y*y;
    float z = 0;
    if (rrxxyy > 0) {
        z = sqrt(rrxxyy);
    }
    z = gpu_ray_march_z(x, y, z, t, planet);
    d_zs[INDEX(i, j, W)] = z;
    d_zs_valid[INDEX(i, j, W)] = (fabs(gpu_sdf(x, y, z, t, planet)) < 1e-2);
}

__global__ void fill_texture_kernel(void *d_pixels, float *d_zs, uint8_t *d_zs_valid, int W, int H, float t, int is_png, int planet) {
    int i = threadIdx.x + 32 * blockIdx.x;
    int j = threadIdx.y + 16 * blockIdx.y;

    if (!d_zs_valid[INDEX(i, j, W)]) {
        if (is_png) {
            ((uint32_t *) d_pixels)[INDEX(i, j, W)] = RGB(0, 0, 0);
            // libattopng_set_pixel((libattopng_t *) pixels, j, i, RGB(0, 0, 0));
        } else {
            ((uint8_t *) d_pixels)[i*W+j] = BLACK;
        }
        return;
    }

    float x = (j - W/2)/(float)H;
    float y = (i - H/2)/(float)H;
    float z = d_zs[INDEX(i, j, W)];

    int color = planet == 0
              ? gpu_earth_color_function(x, y, z, t, is_png)
              : gpu_mars_color_function (x, y, z, t, is_png);

    if (is_png) {
        ((uint32_t *) d_pixels)[j + i * W] = color;
    } else {
        ((uint8_t *) d_pixels)[i*W+j] = color;
    }
}

void cuda_draw_planet(void *pixels, int W, int H, float t, int is_png, int planet, float offset) {
    float   *d_zs;
    uint8_t *d_zs_valid;
    CUDA_CALL( cudaMalloc(&d_zs, W*H*sizeof(float)) );
    CUDA_CALL( cudaMalloc(&d_zs_valid, W*H*sizeof(uint8_t)) );

    cudaMemcpyToSymbol(gpu_perm, perm, 512*sizeof(unsigned char));
    cudaMemcpyToSymbol(GREY_TABLE_GPU, GREY_TABLE, sizeof(GREY_TABLE));
    cudaMemcpyToSymbol(RED_TABLE_GPU, RED_TABLE, sizeof(RED_TABLE));
    cudaMemcpyToSymbol(GREEN_TABLE_GPU, GREEN_TABLE, sizeof(GREEN_TABLE));
    cudaMemcpyToSymbol(BLUE_TABLE_GPU, BLUE_TABLE, sizeof(BLUE_TABLE));

    cudaMemcpyToSymbol(gpu_t_offset, &offset, sizeof(float));

    dim3 blockSize(32, 16);
    dim3 gridSize(H / 32, W / 16); // TODO: check

    make_zs_kernel<<<gridSize, blockSize>>>(d_zs, d_zs_valid, W, H, t, planet);

    if (is_png) {
        uint32_t *d_pixels;
        CUDA_CALL( cudaMalloc(&d_pixels, W*H*sizeof(uint32_t)) );
        fill_texture_kernel<<<gridSize, blockSize>>>(d_pixels, d_zs, d_zs_valid, W, H, t, is_png, planet);
        CUDA_CALL( cudaMemcpy(pixels, d_pixels, W*H*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
        CUDA_CALL( cudaFree(d_pixels) );
    } else {
        uint8_t *d_pixels;
        CUDA_CALL( cudaMalloc(&d_pixels, W*H*sizeof(uint8_t)) );
        fill_texture_kernel<<<gridSize, blockSize>>>(d_pixels, d_zs, d_zs_valid, W, H, t, is_png, planet);
        CUDA_CALL( cudaMemcpy(pixels, d_pixels, W*H*sizeof(uint8_t), cudaMemcpyDeviceToHost) );
        CUDA_CALL( cudaFree(d_pixels) );
    }

    CUDA_CALL( cudaFree(d_zs) );
    CUDA_CALL( cudaFree(d_zs_valid) );
}

