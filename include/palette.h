#ifndef PALETTE_H
#define PALETTE_H

#include <stdlib.h>
#include <stdint.h>

#define BLACK 0
#define WHITE 63

extern const uint8_t GREY_TABLE[64];
extern const uint8_t RED_TABLE[64];
extern const uint8_t GREEN_TABLE[64];
extern const uint8_t BLUE_TABLE[64];
extern uint8_t palette[];

#endif // PALETTE_H