#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float train[][2] = {
    {0,1},
    {1,3},
    {2,5},
    {3,7},
    {4,9},
    {5,11}
};

// defines the length of trainining dataset
#define train_count (sizeof(train)/sizeof(train[0]))

// returns a random number between 0 and 1
float rand_float(void){
    return (float) rand()/(float) RAND_MAX;
}

// Returns the mean squared loss
float cost(float w,float b){
    float loss = 0.0f;
    for (size_t i = 0; i < train_count; i++){
      float y_real = train[i][1];
        float y_pred = w*train[i][0] + b;
        float d = y_pred - y_real;
        loss += d * d;
    }

    return loss/train_count;
}

int main(){
    srand(time(0));

    float w = rand_float() * 10.0f;
    float b = rand_float() * 10.0f;

    float mse = 0.0f;
    int iterations = 1e5;

    float lr = 0.01f;
    float esp = 1e-6;

    for (size_t i = 0; i < train_count; i++){
        float y_real = train[i][1];
        float y_pred = w*train[i][0] + b;
        float d = y_pred - y_real;
        mse += d * d;
        printf("Real : %f, predicted : %f\n",y_real,y_pred);
    }

    printf("\nThe mean square error loss is %f\n\n",mse/train_count);
    printf("_______________Training________________\n");

    for(size_t i = 0; i < iterations; i++){
        float loss = cost(w,b);
        float dw_loss = (cost(w + esp,b) - cost(w,b))/esp;
        float db_loss = (cost(w,b + esp) - cost(w,b))/esp;
        w -= lr * dw_loss;
        b -= lr * db_loss;
        if(i%(iterations/10) == 0) printf("Epoch : [%ld/%d], training_loss : %f\n",i+1,iterations,loss);
        else if(i == iterations-1) printf("Epoch : [%ld/%d], training_loss : %f\n",i+1,iterations,loss);
    }

    printf("\n________Comparing the value of real vs predicted___________\n");
    
    for (size_t i = 0; i < train_count; i++){
        float y_real = train[i][1];
        float y_pred = w*train[i][0] + b;
        printf("Real : %f, predicted : %f\n",y_real,y_pred);
    }
    printf("\nThe mean square error loss at the end of training %f\n\n",cost(w,b));

    printf("The weight of the function is %f\n",w);
    printf("The bias of the function is %f\n",b);

    return 0;
}