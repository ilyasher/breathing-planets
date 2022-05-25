#ifndef NOISE_CUH
#define NOISE_CUH

__constant__ extern unsigned char gpu_perm[512];
__device__ float gpu_noise4( float x, float y, float z, float w );

#endif // NOISE_CUH