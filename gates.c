#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

typedef float sample[3];

// or gate
sample or_train[] = {
    {0,0,0},
    {1,0,1},
    {0,1,1},
    {1,1,1}
};

// and gate
sample and_train[] = {
    {0,0,0},
    {1,0,0},
    {0,1,0},
    {1,1,1}
};

// nand gate
sample nand_train[] = {
    {0,0,1},
    {1,0,1},
    {0,1,1},
    {1,1,0}
};

// change the gate here to check for other gates
sample *train = or_train;
size_t train_count = 4;

// returns a number between 0 and 1
float rand_float(void){
    return (float) rand()/(float) RAND_MAX;
}

// sigmoid function
float sigmoidf(float x){
    return 1.f/(1.f + expf(-x));
}

float cost(float w1, float w2,float b){
    float loss = 0.0f;
    for (size_t i = 0; i < train_count; i++){
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y_pred = w1*x1 + w2*x2 + b;
        y_pred = sigmoidf(y_pred);
        float y_real = train[i][2];
        float d = y_pred-y_real;
        loss += d * d;
    }

    return loss/train_count;
}

int main(){
    srand(time(0));

    float w1 = rand_float();
    float w2 = rand_float();
    float b = rand_float();

    float esp = 1e-2;
    float lr = 0.1f;
    float mse = 0.0f;
    size_t iterations = 2e6;

    for (size_t i = 0; i < train_count; i++){
        float y_real = train[i][2];
        float y_pred = w1*train[i][0] + w1*train[i][1] + b;
        y_pred = sigmoidf(y_pred);
        float d = y_pred - y_real;
        mse += d * d;
        printf("Real : %f, predicted : %f\n",y_real,y_pred);
    }

    printf("\nThe mean square error loss is %f\n\n",mse/train_count);
    printf("_______________Training________________\n");

    for(size_t i = 0; i < iterations; i++){
        float loss = cost(w1,w2,b);
        float dw1 = (cost(w1 + esp,w2,b) - loss)/esp;
        float dw2 = (cost(w1,w2 + esp,b) - loss)/esp;
        float db = (cost(w1,w2, b+esp) - loss)/esp;
        w1 -= lr * dw1;
        w2 -= lr * dw2;
        b  -= lr * db;
        if(i%(iterations/10) == 0) printf("Epoch : [%ld/%ld], training_loss : %f\n",i+1,iterations,loss);
        else if(i == iterations-1) printf("Epoch : [%ld/%ld], training_loss : %f\n",i+1,iterations,loss);
    }

    printf("\n________Comparing the value of real vs predicted___________\n");
    
    for (size_t i = 0; i < train_count; i++){
        float y_real = train[i][2];
        float y_pred = w1*train[i][0] + w1*train[i][1] +b;
        y_pred = sigmoidf(y_pred);
        printf("Real : %f, predicted : %f\n",y_real,y_pred);
    }
    printf("\nThe mean square error loss at the end of training %f\n\n",cost(w1,w2,b));

    printf("The weights of the function are w1 : %f and w2 : %f\n ",w1,w2);
    printf("The bias of the function is %f\n",b);

    return 0;
}