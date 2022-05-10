#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#include "noise.h"

int main(int argc, char *argv[])
{
    srand(time(0));
    double r = 3;

    for (int t = 0; t < 10; t++) {
        for (int i = -20; i < 20; i++) {
            for (int j = -40; j < 40; j++) {
                double x = i / 5.;
                double y = j / 10.;
                double xxyy = x*x+y*y;
                if (xxyy >= r*r) {
                    printf(" ");
                    continue;
                }
                double z = sqrt(r*r-xxyy);
                double n = noise3(x, y, z+0.1*t);
                // printf("%f %f %f\n", x, y, n);
                if (n > 0) {
                    printf("#");
                } else {
                    printf(".");
                }
            }
            printf("\n");
        }

        sleep(1);
    }

	return 0;
}