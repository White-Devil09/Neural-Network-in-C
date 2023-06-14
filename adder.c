#include <time.h>
#define NN_IMPLEMENTATION
#include "nn.h"

#define BITS 1

int main(){
    srand(time(0));

    size_t n = (1<<BITS);
    size_t rows = n*n;
    Mat ti = mat_alloc(rows, 2*BITS);
    Mat to = mat_alloc(rows, BITS + 1);

    for (size_t i = 0; i < ti.rows; i++){
        size_t x = i/n;
        size_t y = i%n;
        size_t z = x+y;
   
        for (size_t j = 0; j < BITS; j++){
            MAT_AT(ti,i,j) = (x>>j)&1;
            MAT_AT(ti,i,j +BITS) = (y>>j)&1;
            MAT_AT(to,i,j) = (z>>j)&1;
        }
        MAT_AT(to,i,BITS) = z>=n;
    }

    size_t arch[] = {2*BITS,4*BITS, BITS+1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn,0,1);

    float lr = 0.1f;
    size_t iterations = 1e6;

    printf("cost  = %f\n",nn_cost(nn,ti,to));

    printf("_______________Training________________\n");

    for (size_t i = 0; i < iterations; i++){
        nn_backprop(nn,g,ti,to);
        nn_weight_update(nn,g,lr);
        if(i%(iterations/10) == 0 || i == iterations-1) printf("Epoch : [%ld/%ld], training_loss : %f\n",i+1,iterations,nn_cost(nn,ti,to));
    }

    printf("\nLoss at the end of training %f\n",nn_cost(nn,ti,to));

    printf("\nWeights and biases of neural network\n");
    NN_PRINT(nn);

    for (size_t x = 0; x < n; x++){
        for (size_t y = 0; y < n; y++){
            printf("%ld + %ld = ",x,y);
            for (size_t j = 0; j < BITS; j++){
                MAT_AT(NN_INPUT(nn),0,j) = (x>>j)&1;
                MAT_AT(NN_INPUT(nn),0,j+BITS) = (y>>j)&1;
            }
            nn_forward(nn);
            if(MAT_AT(NN_OUTPUT(nn),0,BITS)>0.5f){
                printf("Overflown\n");
            }
            else{
                size_t z = 0;
                for (size_t j = 0; j < BITS; j++){
                    size_t bit = MAT_AT(NN_OUTPUT(nn),0,j)>0.5f;
                    z |= bit>>j;
                }
                printf("%ld\n",z); 
            }
        }
    }
    
    
    return 0;
}