#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

typedef float sample[3];

typedef struct {
    float or_w1;
    float or_w2;
    float or_b;

    float nand_w1;
    float nand_w2;
    float nand_b;

    float and_w1;
    float and_w2;
    float and_b;
}xor;

// returns a number between 0 and 1
float rand_float(void){
    return (float) rand()/(float) RAND_MAX;
}

// sigmoid function
float sigmoidf(float x){
    return 1.f/(1.f + expf(-x));
}

// forward pass 
float forward(xor m, float x, float y){
    /* hidden layer 1*/
    float a = sigmoidf(m.or_w1*x + m.or_w2*y + m.or_b);
    float b = sigmoidf(m.nand_w1*x + m.nand_w2*y + m.nand_b);

    /* output layer*/
    return sigmoidf( a * m.and_w1 + b * m.and_w2 + m.and_b);
}

// xor gate
sample xor_train[] = {
    {0,0,0},
    {1,0,1},
    {0,1,1},
    {1,1,0}
};

sample *train = xor_train;
size_t train_count = 4;

// cost function
float cost(xor m){
    float loss = 0.0f;
    for (size_t i = 0; i < train_count; i++){
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y_pred = forward(m, x1, x2);
        float y_real = train[i][2];
        float d = y_pred-y_real;
        loss += d * d;
    }

    return loss/train_count;
}

// randomly initializing weights 
xor rand_xor(void){
    xor m;
    m.or_w1 = rand_float();
    m.or_w2 = rand_float();
    m.or_b = rand_float();

    m.nand_w1 = rand_float();
    m.nand_w2 = rand_float();
    m.nand_b = rand_float();

    m.and_w1 = rand_float();
    m.and_w2 = rand_float();
    m.and_b = rand_float();

    return m;
}

// weight updation
xor finite_diff(xor m,float esp){
    xor g;
    float c = cost(m);
    float saved;

    saved = m.or_w1;
    m.or_w1 += esp;
    g.or_w1 = (cost(m) - c)/esp;
    m.or_w1 = saved;

    saved = m.or_w2;
    m.or_w2 += esp;
    g.or_w2 = (cost(m) - c)/esp;
    m.or_w2 = saved;

    saved = m.or_b;
    m.or_b += esp;
    g.or_b = (cost(m) - c)/esp;
    m.or_b = saved;

    saved = m.nand_w1;
    m.nand_w1 += esp;
    g.nand_w1 = (cost(m) - c)/esp;
    m.nand_w1 = saved;

    saved = m.nand_w2;
    m.nand_w2 += esp;
    g.nand_w2 = (cost(m) - c)/esp;
    m.nand_w2 = saved;

    saved = m.nand_b;
    m.nand_b += esp;
    g.nand_b = (cost(m) - c)/esp;
    m.nand_b = saved;

    saved = m.and_w1;
    m.and_w1 += esp;
    g.and_w1 = (cost(m) - c)/esp;
    m.and_w1 = saved;

    saved = m.and_w2;
    m.and_w2 += esp;
    g.and_w2 = (cost(m) - c)/esp;
    m.and_w2 = saved;

    saved = m.and_b;
    m.and_b += esp;
    g.and_b = (cost(m) - c)/esp;
    m.and_b = saved;

    return g;
}

xor weight_updation(xor m, xor g,float lr){
    m.or_w1 -= lr * g.or_w1;
    m.or_w2 -= lr * g.or_w2;
    m.or_b -= lr * g.or_b;

    m.nand_w1 -= lr * g.nand_w1;
    m.nand_w2 -= lr * g.nand_w2;
    m.nand_b -= lr * g.nand_b;

    m.and_w1 -= lr * g.and_w1;
    m.and_w2 -= lr * g.and_w2;
    m.and_b -= lr * g.and_b;

    return m;
}

int main(){
    srand(time(0));

    xor m = rand_xor();

    float esp = 1e-2;
    float lr = 0.1f;
    size_t iterations = 2e6;

    printf("_______________Training________________\n");

    for(size_t i = 0; i < iterations; i++){
        float loss = cost(m);
        xor g = finite_diff(m,esp);
        m = weight_updation(m,g,lr);
        if(i%(iterations/10) == 0 || i == iterations-1) printf("Epoch : [%ld/%ld], training_loss : %f\n",i+1,iterations,loss);
    }

    printf("\n________Comparing the value of real vs predicted___________\n");
    
    for (size_t i = 0; i < train_count; i++){
        float y_real = train[i][2];
        float y_pred = forward(m,train[i][0],train[i][1]);
        printf("Real : %f, predicted : %f\n",y_real,y_pred);
    }
    printf("\nThe mean square error loss at the end of training %f\n\n",cost(m));

    return 0;
}