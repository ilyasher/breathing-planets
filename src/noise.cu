#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "noise.cuh"

#define FADE(t) ( t * t * t * ( t * ( t * 6 - 15 ) + 10 ) )
#define FASTFLOOR(x) ( ((int)(x)<(x)) ? ((int)x) : ((int)x-1 ) )
#define LERP(t, a, b) ((a) + (t)*((b)-(a)))

__constant__ unsigned char gpu_perm[512];

__device__ static float grad4( int hash, float x, float y, float z, float t ) {
    int h = hash & 31;      // Convert low 5 bits of hash code into 32 simple
    float u = h<24 ? x : y; // gradient directions, and compute dot product.
    float v = h<16 ? y : z;
    float w = h<8 ? z : t;
    return ((h&1)? -u : u) + ((h&2)? -v : v) + ((h&4)? -w : w);
}
__device__ float gpu_noise4( float x, float y, float z, float w )
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

    nxyz0 = grad4(gpu_perm[ix0 + gpu_perm[iy0 + gpu_perm[iz0 + gpu_perm[iw0]]]], fx0, fy0, fz0, fw0);
    nxyz1 = grad4(gpu_perm[ix0 + gpu_perm[iy0 + gpu_perm[iz0 + gpu_perm[iw1]]]], fx0, fy0, fz0, fw1);
    nxy0 = LERP( q, nxyz0, nxyz1 );

    nxyz0 = grad4(gpu_perm[ix0 + gpu_perm[iy0 + gpu_perm[iz1 + gpu_perm[iw0]]]], fx0, fy0, fz1, fw0);
    nxyz1 = grad4(gpu_perm[ix0 + gpu_perm[iy0 + gpu_perm[iz1 + gpu_perm[iw1]]]], fx0, fy0, fz1, fw1);
    nxy1 = LERP( q, nxyz0, nxyz1 );

    nx0 = LERP ( r, nxy0, nxy1 );

    nxyz0 = grad4(gpu_perm[ix0 + gpu_perm[iy1 + gpu_perm[iz0 + gpu_perm[iw0]]]], fx0, fy1, fz0, fw0);
    nxyz1 = grad4(gpu_perm[ix0 + gpu_perm[iy1 + gpu_perm[iz0 + gpu_perm[iw1]]]], fx0, fy1, fz0, fw1);
    nxy0 = LERP( q, nxyz0, nxyz1 );

    nxyz0 = grad4(gpu_perm[ix0 + gpu_perm[iy1 + gpu_perm[iz1 + gpu_perm[iw0]]]], fx0, fy1, fz1, fw0);
    nxyz1 = grad4(gpu_perm[ix0 + gpu_perm[iy1 + gpu_perm[iz1 + gpu_perm[iw1]]]], fx0, fy1, fz1, fw1);
    nxy1 = LERP( q, nxyz0, nxyz1 );

    nx1 = LERP ( r, nxy0, nxy1 );

    n0 = LERP( t, nx0, nx1 );

    nxyz0 = grad4(gpu_perm[ix1 + gpu_perm[iy0 + gpu_perm[iz0 + gpu_perm[iw0]]]], fx1, fy0, fz0, fw0);
    nxyz1 = grad4(gpu_perm[ix1 + gpu_perm[iy0 + gpu_perm[iz0 + gpu_perm[iw1]]]], fx1, fy0, fz0, fw1);
    nxy0 = LERP( q, nxyz0, nxyz1 );

    nxyz0 = grad4(gpu_perm[ix1 + gpu_perm[iy0 + gpu_perm[iz1 + gpu_perm[iw0]]]], fx1, fy0, fz1, fw0);
    nxyz1 = grad4(gpu_perm[ix1 + gpu_perm[iy0 + gpu_perm[iz1 + gpu_perm[iw1]]]], fx1, fy0, fz1, fw1);
    nxy1 = LERP( q, nxyz0, nxyz1 );

    nx0 = LERP ( r, nxy0, nxy1 );

    nxyz0 = grad4(gpu_perm[ix1 + gpu_perm[iy1 + gpu_perm[iz0 + gpu_perm[iw0]]]], fx1, fy1, fz0, fw0);
    nxyz1 = grad4(gpu_perm[ix1 + gpu_perm[iy1 + gpu_perm[iz0 + gpu_perm[iw1]]]], fx1, fy1, fz0, fw1);
    nxy0 = LERP( q, nxyz0, nxyz1 );

    nxyz0 = grad4(gpu_perm[ix1 + gpu_perm[iy1 + gpu_perm[iz1 + gpu_perm[iw0]]]], fx1, fy1, fz1, fw0);
    nxyz1 = grad4(gpu_perm[ix1 + gpu_perm[iy1 + gpu_perm[iz1 + gpu_perm[iw1]]]], fx1, fy1, fz1, fw1);
    nxy1 = LERP( q, nxyz0, nxyz1 );

    nx1 = LERP ( r, nxy0, nxy1 );

    n1 = LERP( t, nx0, nx1 );

    return 0.87f * ( LERP( s, n0, n1 ) );
}