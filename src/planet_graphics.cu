#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// #include "noise.h"
// #include "libattopng.h"
// #include "gifenc.h"
// #include "palette.h"

// #include "mars.h"
// #include "earth.h"

#include "helper_cuda.h"
#include "planet_graphics.cuh"

#define BW 1024

#define HEIGHT_AMP 0.1

#define R 0.45
#define RGBA(r, g, b, a) ((r) | ((g) << 8) | ((b) << 16) | ((a) << 24))
#define RGB(r, g, b) RGBA(r, g, b, 0xff)
#define CAPAT(c, min, max) ((c) < (min) ? (min) : ((c) > (max) ? (max) : (c)))

#define WATER_LEVEL -0.005
#define EARTH_WATER_LEVEL WATER_LEVEL
#define EARTH_SNOW_LEVEL 0.05

#define ROTATION_FREQ -0.7

#define FADE(t) ( t * t * t * ( t * ( t * 6 - 15 ) + 10 ) )

#define FASTFLOOR(x) ( ((int)(x)<(x)) ? ((int)x) : ((int)x-1 ) )
#define LERP(t, a, b) ((a) + (t)*((b)-(a)))
__device__ static unsigned char perm[] = {151,160,137,91,90,15,
  131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
  190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
  88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
  77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
  102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
  135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
  5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
  223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
  129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
  251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
  49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
  138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
  151,160,137,91,90,15,
  131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
  190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
  88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
  77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
  102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
  135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
  5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
  223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
  129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
  251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
  49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
  138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
};
__device__ static float grad4( int hash, float x, float y, float z, float t ) {
    int h = hash & 31;      // Convert low 5 bits of hash code into 32 simple
    float u = h<24 ? x : y; // gradient directions, and compute dot product.
    float v = h<16 ? y : z;
    float w = h<8 ? z : t;
    return ((h&1)? -u : u) + ((h&2)? -v : v) + ((h&4)? -w : w);
}
__device__ static float gpu_noise4( float x, float y, float z, float w )
{
    int ix0, iy0, iz0, iw0, ix1, iy1, iz1, iw1;
    float fx0, fy0, fz0, fw0, fx1, fy1, fz1, fw1;
    float s, t, r, q;
    float nxyz0, nxyz1, nxy0, nxy1, nx0, nx1, n0, n1;

    ix0 = FASTFLOOR( x ); // Integer part of x
    iy0 = FASTFLOOR( y ); // Integer part of y
    iz0 = FASTFLOOR( z ); // Integer part of y
    iw0 = FASTFLOOR( w ); // Integer part of w
    fx0 = x - ix0;        // Fractional part of x
    fy0 = y - iy0;        // Fractional part of y
    fz0 = z - iz0;        // Fractional part of z
    fw0 = w - iw0;        // Fractional part of w
    fx1 = fx0 - 1.0f;
    fy1 = fy0 - 1.0f;
    fz1 = fz0 - 1.0f;
    fw1 = fw0 - 1.0f;
    ix1 = ( ix0 + 1 ) & 0xff;  // Wrap to 0..255
    iy1 = ( iy0 + 1 ) & 0xff;
    iz1 = ( iz0 + 1 ) & 0xff;
    iw1 = ( iw0 + 1 ) & 0xff;
    ix0 = ix0 & 0xff;
    iy0 = iy0 & 0xff;
    iz0 = iz0 & 0xff;
    iw0 = iw0 & 0xff;

    q = FADE( fw0 );
    r = FADE( fz0 );
    t = FADE( fy0 );
    s = FADE( fx0 );

    nxyz0 = grad4(perm[ix0 + perm[iy0 + perm[iz0 + perm[iw0]]]], fx0, fy0, fz0, fw0);
    nxyz1 = grad4(perm[ix0 + perm[iy0 + perm[iz0 + perm[iw1]]]], fx0, fy0, fz0, fw1);
    nxy0 = LERP( q, nxyz0, nxyz1 );

    nxyz0 = grad4(perm[ix0 + perm[iy0 + perm[iz1 + perm[iw0]]]], fx0, fy0, fz1, fw0);
    nxyz1 = grad4(perm[ix0 + perm[iy0 + perm[iz1 + perm[iw1]]]], fx0, fy0, fz1, fw1);
    nxy1 = LERP( q, nxyz0, nxyz1 );

    nx0 = LERP ( r, nxy0, nxy1 );

    nxyz0 = grad4(perm[ix0 + perm[iy1 + perm[iz0 + perm[iw0]]]], fx0, fy1, fz0, fw0);
    nxyz1 = grad4(perm[ix0 + perm[iy1 + perm[iz0 + perm[iw1]]]], fx0, fy1, fz0, fw1);
    nxy0 = LERP( q, nxyz0, nxyz1 );

    nxyz0 = grad4(perm[ix0 + perm[iy1 + perm[iz1 + perm[iw0]]]], fx0, fy1, fz1, fw0);
    nxyz1 = grad4(perm[ix0 + perm[iy1 + perm[iz1 + perm[iw1]]]], fx0, fy1, fz1, fw1);
    nxy1 = LERP( q, nxyz0, nxyz1 );

    nx1 = LERP ( r, nxy0, nxy1 );

    n0 = LERP( t, nx0, nx1 );

    nxyz0 = grad4(perm[ix1 + perm[iy0 + perm[iz0 + perm[iw0]]]], fx1, fy0, fz0, fw0);
    nxyz1 = grad4(perm[ix1 + perm[iy0 + perm[iz0 + perm[iw1]]]], fx1, fy0, fz0, fw1);
    nxy0 = LERP( q, nxyz0, nxyz1 );

    nxyz0 = grad4(perm[ix1 + perm[iy0 + perm[iz1 + perm[iw0]]]], fx1, fy0, fz1, fw0);
    nxyz1 = grad4(perm[ix1 + perm[iy0 + perm[iz1 + perm[iw1]]]], fx1, fy0, fz1, fw1);
    nxy1 = LERP( q, nxyz0, nxyz1 );

    nx0 = LERP ( r, nxy0, nxy1 );

    nxyz0 = grad4(perm[ix1 + perm[iy1 + perm[iz0 + perm[iw0]]]], fx1, fy1, fz0, fw0);
    nxyz1 = grad4(perm[ix1 + perm[iy1 + perm[iz0 + perm[iw1]]]], fx1, fy1, fz0, fw1);
    nxy0 = LERP( q, nxyz0, nxyz1 );

    nxyz0 = grad4(perm[ix1 + perm[iy1 + perm[iz1 + perm[iw0]]]], fx1, fy1, fz1, fw0);
    nxyz1 = grad4(perm[ix1 + perm[iy1 + perm[iz1 + perm[iw1]]]], fx1, fy1, fz1, fw1);
    nxy1 = LERP( q, nxyz0, nxyz1 );

    nx1 = LERP ( r, nxy0, nxy1 );

    n1 = LERP( t, nx0, nx1 );

    return 0.87f * ( LERP( s, n0, n1 ) );
}


typedef struct vec3_t {
    float x;
    float y;
    float z;
} vec3_t;

__device__ static vec3_t lp = {-.7, .2, .7};


#define INDEX(i, j, W) ((i) * (W) + (j))

// float water_level = -1;
// float (*height_function)(float) = mars_height_function;
// int   (*color_function)(float, float, float, float, int) = mars_color_function;
// float seed = 0;

// Simplex noise added up at a few octaves
__device__ static float gpu_noise(float x, float y, float z, float t) {

    float x_rot = x*cos(ROTATION_FREQ * t) - z*sin(ROTATION_FREQ * t);
    float z_rot = x*sin(ROTATION_FREQ * t) + z*cos(ROTATION_FREQ * t);
    float y_rot = y /*+ seed*/;

    float power = 1;
    float n = 0;
    for (int freq = 2; freq < 256; freq *= 2) {
        n += power * gpu_noise4(x_rot*freq, y_rot*freq, z_rot*freq, t);
        power /= 2.2;
    }

    // n = height_function(n);
    n = n * sqrt(fabs(n));
    return n;
}

__device__ static float gpu_surface_height(float x, float y, float z, float t) {
    float norm = sqrt(x*x+y*y+z*z);
    float height = HEIGHT_AMP*gpu_noise(R*x/norm, R*y/norm, R*z/norm, t);
    return height;
}

__device__ static float gpu_sdf(float x, float y, float z, float t) {
    float norm = sqrt(x*x+y*y+z*z);
    float h = gpu_surface_height(x, y, z, t);
    return norm - (R + (h > WATER_LEVEL ? h : WATER_LEVEL));
}


__device__ static void gpu_normalize(vec3_t* v) {
    float norm = sqrt(v->x*v->x + v->y*v->y + v->z*v->z);
    v->x /= norm;
    v->y /= norm;
    v->z /= norm;
}


__device__ static void gpu_grad_sdf(vec3_t *grad, float x, float y, float z, float t) {
    float delta = 1e-3;
    float p0 = gpu_sdf(x, y, z, t);
    float px = gpu_sdf(x+delta, y, z, t);
    float py = gpu_sdf(x, y+delta, z, t);
    float pz = gpu_sdf(x, y, z+delta, t);
    grad->x = (px - p0) / delta;
    grad->y = (py - p0) / delta;
    grad->z = (pz - p0) / delta;
    gpu_normalize(grad);
}

// Given x, y, find which z is at the planet's surface
// (try find z such that f(x, y, z, t) = 0)
__device__ static float gpu_ray_march_z(float x, float y, float z, float t) {
    for (int i = 0; i < 32; i++) {
        z = z - gpu_sdf(x, y, z, t);
    }
    return z;
}

#define AMBIENT 0.1

// Return number 0-1 representing how much lighting the point has.
__device__ static float gpu_lighting(float x, float y, float z, float t) {

    vec3_t ray = {lp.x * cos(t), lp.y, lp.z -14*sin(t)};
    vec3_t ld = {x-ray.x, y-ray.y, z-ray.z};
    gpu_normalize(&ld);
    for (int i = 0; i < 32; i++) {
        float dist = gpu_sdf(ray.x, ray.y, ray.z, t);
        ray.x += dist*ld.x;
        ray.y += dist*ld.y;
        ray.z += dist*ld.z;
    }
    if ((ray.x-x)*(ray.x-x)+(ray.y-y)*(ray.y-y)+(ray.z-z)*(ray.z-z) > 1e-4) {
        // In the shadow
        return AMBIENT;
    }

    vec3_t grad;
    gpu_grad_sdf(&grad, x, y, z, t);
    // float delta = 1e-3;
    // float p0 = gpu_sdf(1, 0, 0, 0);
    // float p0 = gpu_sdf(ray.x, ray.y, ray.z, t);
    // float p0 = gpu_sdf(x, y, z, t);
    // float px = gpu_sdf(x+delta, y, z, t);
    // float py = gpu_sdf(x, y+delta, z, t);
    // float pz = gpu_sdf(x, y, z+delta, t);
    // grad.x = (px - p0) / delta;
    // grad.y = (py - p0) / delta;
    // grad.z = (pz - p0) / delta;
    // gpu_normalize(&grad);

    float diffuse = -ld.x*grad.x - ld.y*grad.y - ld.z*grad.z;
    diffuse = diffuse > 0 ? diffuse : 0;

    float specular = grad.z > 0 ? grad.z : 0;
    specular = pow(specular, 256);

    return AMBIENT + 0.9 * diffuse;
}

__device__ static int gpu_earth_color_function(float x, float y, float z, float t, int is_png) {
    // float light = gpu_lighting(x, y, z, t);

    float light = 1;
    float n = gpu_surface_height(x, y, z, t);

    if (n < EARTH_WATER_LEVEL) {
        if (is_png) {
            int c = CAPAT((48+n*64) * light, 0, 255);
            return RGB(c, c,  (int) (255*light));
        }
        // int c = 48 + (int)(n*64);
        // return BLUE_TABLE[CAPAT((int)(c*light), 0, 63)];
    }
    if (n + fabs(y / 10) + 0.01 * gpu_noise4(256*x,256*y,256*z,t) < EARTH_SNOW_LEVEL) {
        if (is_png) {
            int c = CAPAT((light * (100 + (int)(n*3000))), 0, 255);
            return RGB(0, c, 0);
        }
        // int c = 16+(int)(n*96);
        // return GREEN_TABLE[CAPAT((int)(c*light), 0, 63)];
    } else {
        if (is_png) {
            int c = CAPAT((light * (100 + (int)(n*3000))), 0, 255);
            return RGB(c, c, c);
        }
        // int c = 63;
        // return GREY_TABLE[(int)(c*light)];
    }
    return RGB(255, 0, 0);
}


__global__ void make_zs_kernel(float *d_zs, uint8_t *d_zs_valid, int W, int H, float t) {
    int i = threadIdx.x + 32 * blockIdx.x;
    int j = threadIdx.y + 32 * blockIdx.y;

    float x = (j - W/2)/(float)H;
    float y = (i - H/2)/(float)H;
    float rrxxyy = R*R-x*x+y*y;
    float z = 0;
    if (rrxxyy > 0) {
        z = sqrt(rrxxyy);
    }
    z = gpu_ray_march_z(x, y, z, t);
    d_zs[INDEX(i, j, W)] = z;
    d_zs_valid[INDEX(i, j, W)] = (fabs(gpu_sdf(x, y, z, t)) < 1e-2);
}


// Do the raytracing for every x, y and fill z values in zs
void cuda_make_zs(float *zs, uint8_t *zs_valid, int W, int H, float t) {
    float   *d_zs;
    uint8_t *d_zs_valid;
    CUDA_CALL( cudaMalloc(&d_zs, W*H*sizeof(float)) );
    CUDA_CALL( cudaMalloc(&d_zs_valid, W*H*sizeof(uint8_t)) );

    dim3 blockSize(32, 32);
    dim3 gridSize(W / 32, H / 32); // TODO: check

    make_zs_kernel<<<gridSize, blockSize>>>(d_zs, d_zs_valid, W, H, t);

    CUDA_CALL( cudaMemcpy(zs, d_zs, W*H*sizeof(float), cudaMemcpyDeviceToHost) );
    CUDA_CALL( cudaMemcpy(zs_valid, d_zs_valid, W*H*sizeof(uint8_t), cudaMemcpyDeviceToHost) );
    CUDA_CALL( cudaFree(d_zs) );
    CUDA_CALL( cudaFree(d_zs_valid) );
}


__global__ void fill_texture_kernel(void *d_pixels, float *d_zs, uint8_t *d_zs_valid, int W, int H, float t, int is_png) {
    int i = threadIdx.x + 32 * blockIdx.x;
    int j = threadIdx.y + 32 * blockIdx.y;


    if (!d_zs_valid[INDEX(i, j, W)]) {
        if (is_png) {
            ((uint32_t *) d_pixels)[INDEX(i, j, W)] = RGB(0, 0, 0);
            // libattopng_set_pixel((libattopng_t *) pixels, j, i, RGB(0, 0, 0));
        } else {
            // ((uint8_t *) d_pixels)[i*W+j] = BLACK;
        }
        return;
    }

    float x = (j - W/2)/(float)H;
    float y = (i - H/2)/(float)H;
    float z = d_zs[INDEX(i, j, W)];
    int color = gpu_earth_color_function(x, y, z, t, is_png);

    // ((uint32_t *) d_pixels)[INDEX(i, j, W)] = RGB(0, 0, 255);
    // #define INDEX(i, j, W) ((i) * (W) + (j))
    // ((uint32_t *) d_pixels)[j + i * W] = RGB(0, 255, 0);

    // return;

    if (is_png) {
        ((uint32_t *) d_pixels)[j + i * W] = color;
        // libattopng_set_pixel((libattopng_t *) pixels, j, i, color);
    } else {
        ((uint8_t *) d_pixels)[i*W+j] = color;
    }
}

void cuda_fill_texture(void *pixels, float *zs, uint8_t *zs_valid, int W, int H, float t, int is_png) {
    float   *d_zs;
    uint8_t *d_zs_valid;
    CUDA_CALL( cudaMalloc(&d_zs, W*H*sizeof(float)) );
    CUDA_CALL( cudaMalloc(&d_zs_valid, W*H*sizeof(uint8_t)) );
    CUDA_CALL( cudaMemcpy(d_zs, zs, W*H*sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy(d_zs_valid, zs_valid, W*H*sizeof(uint8_t), cudaMemcpyHostToDevice) );

    dim3 blockSize(32, 32);
    dim3 gridSize(W / 32, H / 32);

    if (is_png) {
        uint32_t *d_pixels;
        CUDA_CALL( cudaMalloc(&d_pixels, W*H*sizeof(uint32_t)) );
        fill_texture_kernel<<<gridSize, blockSize>>>(d_pixels, d_zs, d_zs_valid, W, H, t, is_png);
        CUDA_CALL( cudaMemcpy(pixels, d_pixels, W*H*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
        CUDA_CALL( cudaFree(d_pixels) );
    } else {
        uint8_t *d_pixels;
        CUDA_CALL( cudaMalloc(&d_pixels, W*H*sizeof(uint8_t)) );
        fill_texture_kernel<<<gridSize, blockSize>>>(d_pixels, d_zs, d_zs_valid, W, H, t, is_png);
        CUDA_CALL( cudaMemcpy(pixels, d_pixels, W*H*sizeof(uint8_t), cudaMemcpyDeviceToHost) );
        CUDA_CALL( cudaFree(d_pixels) );
    }

    CUDA_CALL( cudaFree(d_zs) );
    CUDA_CALL( cudaFree(d_zs_valid) );
}
