#ifndef NN_H_
#define NN_H_

#include <stddef.h>
#include <stdio.h>
#include <math.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc 
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT

#define ARRAY_LEN(xs) sizeof((xs))/sizeof((xs)[0])

float rand_float(void);
float sigmoidf(float x);

typedef struct{
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
}Mat;

#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]

Mat mat_alloc(size_t rows, size_t cols);
void mat_fill(Mat m, float x);
void mat_rand(Mat m,float low, float high);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dst, Mat src);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat a);
void mat_sig(Mat m);
void mat_print(Mat m,const char *name, size_t pad);
#define MAT_PRINT(m) mat_print(m, #m, 0)

typedef struct{
    size_t count;
    Mat *ws;
    Mat *bs;
    Mat *as;
}NN;

#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]

NN nn_alloc(size_t *arch, size_t arch_count);
void nn_zero(NN nn);
void nn_print(NN nn, const char *name);
#define NN_PRINT(nn) nn_print(nn, #nn);
void nn_rand(NN nn, float low, float high);
void nn_forward(NN nn);
float nn_cost(NN nn, Mat in, Mat out);
void nn_fininte_diff(NN nn, NN g, float esp, Mat in, Mat out);
void nn_backprop(NN nn, NN g, Mat in, Mat out);
void nn_weight_update(NN nn, NN g, float lr);

#endif // NN_H_

#ifdef NN_IMPLEMENTATION

float sigmoidf(float x){
    return 1.f/(1.f + expf(-x));
}

float rand_float(void){
    return (float) rand()/(float) RAND_MAX;
}

Mat mat_alloc(size_t rows, size_t cols){
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = NN_MALLOC(sizeof(*m.es)*rows*cols);
    NN_ASSERT(m.es != NULL);

    return m;
}

void mat_dot(Mat dst, Mat a, Mat b){
    NN_ASSERT(a.cols == b.rows);
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == b.cols);

    for (size_t i = 0; i < dst.rows; i++){
        for (size_t j = 0; j < dst.cols; j++){
            MAT_AT(dst,i ,j) = 0;
            for (size_t k = 0; k < a.cols; k++){
                MAT_AT(dst,i ,j) += MAT_AT(a,i,k) * MAT_AT(b,k,j);
            }
        }
    }
}

Mat mat_row(Mat m, size_t row){
    return (Mat){
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .es = &MAT_AT(m, row, 0)
    };
}

void mat_copy(Mat dst, Mat src){
    NN_ASSERT(dst.rows == src.rows);
    NN_ASSERT(dst.cols == src.cols);

    for (size_t i = 0; i < dst.rows; i++){
        for (size_t j = 0; j < dst.cols; j++){
            MAT_AT(dst, i, j) = MAT_AT(src,i,j);
        }  
    }
}


void mat_sum(Mat dst, Mat a){
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == a.cols);

    for (size_t i = 0; i < dst.rows; i++){
        for (size_t j = 0; j < dst.cols; j++){
            MAT_AT(dst,i,j) += MAT_AT(a,i,j);
        }     
    }  
}

void mat_sig(Mat m){
    for (size_t i = 0; i < m.rows; i++){
        for (size_t j = 0; j < m.cols; j++){
            MAT_AT(m,i,j) = sigmoidf(MAT_AT(m,i,j));
        }
    }
}

void mat_print(Mat m,const char *name, size_t pad){
    printf("%*s%s = [\n",(int) pad, "" , name);
    for (size_t i = 0; i < m.rows; i++){
        printf("%*s    ",(int) pad, "");
        for (size_t j = 0; j < m.cols; j++){
            printf("%f  ",MAT_AT(m,i,j));
        }
        printf("\n");
    }
    printf("%*s]\n",(int)pad, "");
    
}

void mat_fill(Mat m, float x){
    for (size_t i = 0; i < m.rows; i++){
        for (size_t j = 0; j < m.cols; j++){
            MAT_AT(m,i,j) = x;
        }
    }
}

void mat_rand(Mat m,float low, float high){
    for (size_t i = 0; i < m.rows; i++){
        for (size_t j = 0; j < m.cols; j++){
            MAT_AT(m,i,j) = rand_float() *(high - low) + low;
        }
    }
}

NN nn_alloc(size_t *arch, size_t arch_count){
    NN_ASSERT(arch_count > 0);
    
    NN nn;
    nn.count = arch_count - 1;

    nn.ws = NN_MALLOC(sizeof(*nn.ws)*nn.count);
    NN_ASSERT(nn.ws != NULL);
    nn.bs = NN_MALLOC(sizeof(*nn.bs)*nn.count);
    NN_ASSERT(nn.bs != NULL);
    nn.as = NN_MALLOC(sizeof(*nn.as)*(nn.count + 1));
    NN_ASSERT(nn.as != NULL);

    nn.as[0] = mat_alloc(1, arch[0]);

    for (size_t i = 1; i < arch_count; i++){
        nn.ws[i-1] = mat_alloc(nn.as[i-1].cols, arch[i]);
        nn.bs[i-1] = mat_alloc(1,arch[i]);
        nn.as[i] = mat_alloc(1,arch[i]);
    }

    return nn;
}

void nn_zero(NN nn){
    for (size_t i = 0; i < nn.count; i++){
        mat_fill(nn.ws[i], 0);
        mat_fill(nn.bs[i], 0);
        mat_fill(nn.as[i], 0);
    }
    mat_fill(nn.as[nn.count],0);
    
}

void nn_print(NN nn, const char *name){
    char buf[256];
    printf("%s = [ \n",name);
    for (size_t i = 0; i < nn.count; i++){
        snprintf(buf, sizeof(buf), "ws[%zu]",i);
        mat_print(nn.ws[i],buf, 8);
        printf("\n");
        snprintf(buf, sizeof(buf), "bs[%zu]",i);
        mat_print(nn.bs[i],buf, 8);
        printf("\n");
    }
    printf("]\n");
}

void nn_rand(NN nn, float low, float high){
    for (size_t i = 0; i < nn.count; i++){
        mat_rand(nn.ws[i],low, high);
        mat_rand(nn.bs[i],low, high);
    }
}

void nn_forward(NN nn){
    for (size_t i = 0; i < nn.count; i++){
        mat_dot(nn.as[i+1], nn.as[i], nn.ws[i]);
        mat_sum(nn.as[i+1], nn.bs[i]);
        mat_sig(nn.as[i+1]);
    }  
}

float nn_cost(NN nn, Mat in, Mat out){
    NN_ASSERT(in.rows == out.rows);
    NN_ASSERT(out.cols == NN_OUTPUT(nn).cols);

    float c = 0;
    for (size_t i = 0; i < in.rows; i++){
        Mat x = mat_row(in, i);
        Mat y = mat_row(out, i);

        mat_copy(NN_INPUT(nn), x);
        nn_forward(nn);

        for (size_t j = 0; j < out.cols; j++){
            float d = MAT_AT(NN_OUTPUT(nn),0,j) - MAT_AT(y,0,j);
            c += d*d;
        }   
    }

    return c/in.rows;
    
}

void nn_backprop(NN nn, NN g, Mat in, Mat out){
    NN_ASSERT(in.rows == out.rows);
    NN_ASSERT(NN_OUTPUT(nn).cols == out.cols);

    nn_zero(g);

    for (size_t i = 0; i < in.rows; i++){
        mat_copy(NN_INPUT(nn),mat_row(in, i));
        nn_forward(nn);

        for (size_t j = 0; j <= nn.count; j++){
            mat_fill(g.as[j], 0);
        }
        
        for (size_t j = 0; j < out.cols; j++){
            MAT_AT(NN_OUTPUT(g), 0, j) = MAT_AT(NN_OUTPUT(nn),0,j) - MAT_AT(out, i,j);
        }
        
        for (size_t l = nn.count; l > 0; l--){
            for (size_t j = 0; j < nn.as[l].cols; j++){
                float a = MAT_AT(nn.as[l],0,j);
                float da = MAT_AT(g.as[l],0,j);
                MAT_AT(g.bs[l-1],0,j) += 2*da*a*(1-a);
                for (size_t k = 0; k < nn.as[l-1].cols; k++){
                    float pa = MAT_AT(nn.as[l-1], 0, k);
                    float w = MAT_AT(nn.ws[l-1],k,j);
                
                    MAT_AT(g.ws[l-1],k ,j) += 2*da*a*(1-a)*pa;
                    MAT_AT(g.as[l-1],0 ,k) += 2*da*a*(1-a)*w;
                }
            }
        }
    }

    for (size_t i = 0; i < g.count; i++){
        for (size_t j = 0; j < g.ws[i].rows; j++){
            for (size_t k = 0; k < g.ws[i].cols; k++){
                MAT_AT(g.ws[i], j, k) /= in.rows;
            }
        }

        for (size_t j = 0; j < g.bs[i].rows; j++){
            for (size_t k = 0; k < g.bs[i].cols; k++){
                MAT_AT(g.bs[i], j, k) /= in.rows;
            } 
        }
    }
}

void nn_fininte_diff(NN nn, NN g, float esp, Mat in, Mat out){
    float saved;
    float c = nn_cost(nn, in, out);

    for (size_t i = 0; i < nn.count; i++){
        for (size_t j = 0; j < nn.ws[i].rows; j++){
            for (size_t k = 0; k < nn.ws[i].cols; k++){
                saved = MAT_AT(nn.ws[i], j, k);
                MAT_AT(nn.ws[i], j, k) += esp;
                MAT_AT(g.ws[i], j, k) = (nn_cost(nn, in,out) - c)/esp;
                MAT_AT(nn.ws[i], j, k) = saved;
            }
        }

        for (size_t j = 0; j < nn.bs[i].rows; j++){
            for (size_t k = 0; k < nn.bs[i].cols; k++){
                saved = MAT_AT(nn.bs[i], j, k);
                MAT_AT(nn.bs[i], j, k) += esp;
                MAT_AT(g.bs[i], j, k) = (nn_cost(nn, in,out) - c)/esp;
                MAT_AT(nn.bs[i], j, k) = saved;
            }
        }
    }   
}

void nn_weight_update(NN nn, NN g, float lr){
    for (size_t i = 0; i < nn.count; i++){
        for (size_t j = 0; j < nn.ws[i].rows; j++){
            for (size_t k = 0; k < nn.ws[i].cols; k++){
                MAT_AT(nn.ws[i], j, k) -= lr * MAT_AT(g.ws[i], j, k);
            }
        }

        for (size_t j = 0; j < nn.bs[i].rows; j++){
            for (size_t k = 0; k < nn.bs[i].cols; k++){
                MAT_AT(nn.bs[i], j, k) -= lr * MAT_AT(g.bs[i], j, k);
            }
        }
    }
}

#endif // NN_IMPLEMENTATION