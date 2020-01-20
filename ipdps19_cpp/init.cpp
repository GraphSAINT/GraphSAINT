#include "global.h"
#include "init.h"
#include <math.h>
#include <stdlib.h>
#include <omp.h>

void init_glorot(s_data2d_ds &weight) {
    t_data range=sqrt(6./(weight.dim1+weight.dim2));
    int range_int=(int)(range*10000);
    srand(0);
    #pragma omp parallel for
    for (int i=0;i<weight.dim1;i++) {
        for (int j=0;j<weight.dim2;j++) {
            t_data temp=(t_data)(rand()%(2*range_int)-range_int);
            weight.arr[i*weight.dim2+j]=temp/10000.;
        }
    }
}

void init_zero(s_data1d_ds &bias) {
    #pragma omp parallel for
    for (int i=0;i<bias.dim1;i++) {
        bias.arr[i]=0.0;
    }
}
