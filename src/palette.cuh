#ifndef PALETTE_CUH
#define PALETTE_CUH

#include "palette.h"

__constant__ uint8_t GREY_TABLE_GPU[64];
__constant__ uint8_t RED_TABLE_GPU[64];
__constant__ uint8_t GREEN_TABLE_GPU[64];
__constant__ uint8_t BLUE_TABLE_GPU[64];

#endif // PALETTE_CUH
