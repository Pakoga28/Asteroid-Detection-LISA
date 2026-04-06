
#define _USE_MATH_DEFINES
#define LINE_MAX 150
#include "lisa.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

int R = 1; // orbit radius in AU
double ecc = 0.00965; // eccentricity of orbit
int f_m = 1; // modulation frequency (1/yr)
// these give the initial ecliptic longitude and orientation of the constellation
double lambda = 0;
double kappa = 0;

char* model(double t, int n) {
    double alpha = 2 * M_PI * f_m * t + kappa;
    double beta = 2 * M_PI * n / 3 + lambda;
    double x = R * cos(alpha) + 0.5 * ecc * R * (cos(2 * alpha - beta) - 3 * cos(beta)) + 0.125 * pow(ecc, 2) * R * (3 * cos(3 * alpha - 2 * beta) - 10 * cos(alpha) - 5 * cos(alpha - 2 * beta));
    double y = R * sin(alpha) + 0.5 * ecc * R * (sin(2 * alpha - beta) - 3 * sin(beta)) + 0.125 * pow(ecc, 2) * R * (3 * sin(3 * alpha - 2 * beta) - 10 * sin(alpha) + 5 * sin(alpha - 2 * beta));
    double z = -sqrt(3) * ecc * R * cos(alpha - beta) + sqrt(3) * pow(ecc, 2) * R * (pow(cos(alpha - beta), 2) + 2 * pow(sin(alpha - beta), 2));

    // write position data to string for output
    char tempText[LINE_MAX];
    char* rtnStr = (char*)malloc(LINE_MAX);
    snprintf(tempText, sizeof(tempText), "%.10lf,%.10lf,%.10lf", x, y, z);
    snprintf(rtnStr, strlen(tempText), "%s", tempText);
    return rtnStr;
}

int main() {
    printf("Data Creation Started...\n");
    FILE* file = fopen("./plots/export_lisa.csv", "w");
    // data format: "x0,y0,z0,x1,y1,z1,x2,y2,z2\n"
    // time is in years
    double t_max = 12;
    double dt = 1.0 / ((double)365.25 * 24);
    int N = (int)floor(t_max / dt);

    // create, initialize and fill a time array
    double* times = (double*)malloc(N * sizeof(double));
    times[0] = 0;
    for (int i = 1; i < N; i++) {
        times[i] = times[i - 1] + dt;
    }

    // calculate the position values for each time and each n
    for (int i = 0; i < N; i++) {
        for (int n = 0; n < 3; n++) {
            char* data_i_n = model(times[i], n);
            // write position data to file, new line after third set
            if (n < 2) {
                fprintf(file, "%s,", data_i_n);
            }
            else {
                fprintf(file, "%s\n", data_i_n);
            }
            free(data_i_n);
        }
    }
    free(times);
    fclose(file);
    printf("Data Creation Finished...\n");
    return 0;
}
