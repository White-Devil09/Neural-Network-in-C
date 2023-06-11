#include <time.h>
#define NN_IMPLEMENTATION
#include "nn.h"

float td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0
};

int main(){
    srand(time(0));

    size_t stride = 3;
    size_t n = sizeof(td)/sizeof(td[0])/stride;

    float esp = 1e-2;
    float lr = 0.1f;
    size_t iterations = 2e6;

    Mat ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = td
    };

    Mat to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .es = td + 2
    };

    size_t arch[] = {2,2,1};
    NN nn = nn_alloc(arch,ARRAY_LEN(arch));
    NN g = nn_alloc(arch,ARRAY_LEN(arch));
    nn_rand(nn, 0,1);

    printf("cost  = %f\n",nn_cost(nn,ti,to));

    printf("_______________Training________________\n");

    for (size_t i = 0; i < iterations; i++){
        nn_fininte_diff(nn,g,esp,ti,to);
        nn_weight_update(nn,g,lr);
        if(i%(iterations/10) == 0 || i == iterations-1) printf("Epoch : [%ld/%ld], training_loss : %f\n",i+1,iterations,nn_cost(nn,ti,to));
    }

    printf("\nLoss at the end of training %f\n",nn_cost(nn,ti,to));

    printf("\n________predicted values___________\n");
    for (size_t i = 0; i < 2; i++){
        for (size_t j = 0; j < 2; j++){
            MAT_AT(NN_INPUT(nn),0,0) = i;
            MAT_AT(NN_INPUT(nn),0,1) = j;
            nn_forward(nn);
            printf("%ld ^ %ld = %f\n",i,j,MAT_AT(NN_OUTPUT(nn),0,0));
        }
    }

    printf("\nWeights and biases of neural network\n");
    NN_PRINT(nn);
    
    return 0;
}