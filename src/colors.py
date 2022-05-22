
print("#pragma once")
print("const uint8_t BLACK = 0;")
print("const uint8_t WHITE = 63;")

print("const uint8_t GREY_TABLE[64] = {", end='')
for i in range(64):
    print(f'{i},', end='')
print('};')
print("const uint8_t RED_TABLE[64] = {", end='')
for i in range(64, 128):
    print(f'{i},', end='')
print('};')
print("const uint8_t GREEN_TABLE[64] = {", end='')
for i in range(128, 192):
    print(f'{i},', end='')
print('};')
print("const uint8_t BLUE_TABLE[64] = {", end='')
for i in range(192, 256):
    print(f'{i},', end='')
print('};')

print("uint8_t palette[] = {")
#BLACK to WHITE
for i in range(64):
    c = int(255 * (i/63))
    print(f'{c}, {c}, {c},')

#RED
for i in range(32):
    c = int(255 * (i/32))
    print(f'{c}, 0, 0,')
for i in range(32):
    c = int(255 * (i/32))
    print(f'255, {c}, {c},')

#GREEN
for i in range(32):
    c = int(255 * (i/32))
    print(f'0, {c}, 0,')
for i in range(32):
    c = int(255 * (i/32))
    print(f'{c}, 255, {c},')

#BLUE
for i in range(32):
    c = int(255 * (i/32))
    print(f'0, 0, {c},')
for i in range(32):
    c = int(255 * (i/32))
    print(f'{c}, {c}, 255,')


print("};")

