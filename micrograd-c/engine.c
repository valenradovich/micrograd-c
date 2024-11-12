#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "engine.h"

// initialize value method
Value *init_value(double data, Value **children, int num_children, const char *op) {
    Value *v = malloc(sizeof(Value)); 
    if (v == NULL) {
        fprintf(stderr, "failed to allocate memory for Value object\n");
        return NULL;
    }
    v->data = data;
    v->grad = 0.0;
    v->_backward = NULL;
    v->_prev = malloc(num_children * sizeof(Value*));
    memcpy(v->_prev, children, num_children * sizeof(Value*));
    v->_prev_count = num_children;
    v->_op = strdup(op);
    return v;
}

// addition operation and backward
Value *add(Value *a, Value *b) {
    Value *children[] = {a, b};
    Value *out = init_value(a->data + b->data, children, 2, "+");

    out->_backward = add_backward;
    return out;
}

void add_backward(Value *v) {
    v->_prev[0]->grad += v->grad;
    v->_prev[1]->grad += v->grad;
}

// multiplication operation and backward
Value *mul(Value *a, Value *b) {
    Value *children[] = {a, b};
    Value *out = init_value(a->data * b->data, children, 2, "*");

    out->_backward = mul_backward;
    return out;
}

void mul_backward(Value *v) {
    v->_prev[0]->grad += v->_prev[1]->data * v->grad;
    v->_prev[1]->grad += v->_prev[0]->data * v->grad;
}

// power operation and backward
Value *power(Value *a, double poww) {
    Value *children[] = {a};
    char power_str[20];
    snprintf(power_str, sizeof(power_str), "**%.2f", poww);
    Value *out = init_value(pow(a->data, poww), children, 1, power_str);

    out->_backward = power_backward;
    return out;
}

void power_backward(Value* v) {
    double power;
    sscanf(v->_op + 2, "%lf", &power);
    v->_prev[0]->grad += power * pow(v->_prev[0]->data, power - 1) * v->grad;
}

// ReLU activation function and backward
Value* relu(Value* a) {
    Value* children[] = {a};
    Value* out = init_value(a->data > 0 ? a->data : 0, children, 1, "ReLU");

    out->_backward = relu_backward;
    return out;
}

void relu_backward(Value* v) {
    v->_prev[0]->grad += (v->_prev[0]->data > 0) ? v->grad : 0;
}

// negation operation
Value* neg(Value* a) {
    return mul(a, init_value(-1, NULL, 0, ""));
}

// subtraction operation
Value* sub(Value* a, Value* b) {
    return add(a, neg(b));
}

// division operation
Value* division(Value* a, Value* b) {
    return mul(a, power(b, -1));
}

// builds topological order of the computation graph
void build_topo(Value* v, Value*** sorted, int* size, int* capacity) {
    // recursively process children
    for (int i = 0; i < v->_prev_count; i++) {
        if (v->_prev[i]->_backward) {
            build_topo(v->_prev[i], sorted, size, capacity);
        }
    }
    
    // check if node is already in sorted list
    for (int i = 0; i < *size; i++) {
        if ((*sorted)[i] == v) return;
    }
    
    // expand capacity if needed
    if (*size >= *capacity) {
        *capacity *= 2;
        *sorted = realloc(*sorted, *capacity * sizeof(Value*));
    }
    
    // add node to sorted list
    (*sorted)[(*size)++] = v;
}

// backward pass (backpropagation)
void backward(Value* v) {
    int size = 0;
    int capacity = 10;
    Value** topo = malloc(capacity * sizeof(Value*));
    build_topo(v, &topo, &size, &capacity);

    // set gradient of output to 1
    v->grad = 1;
    
    // backward pass in reverse topological order
    for (int i = size - 1; i >= 0; i--) {
        if (topo[i]->_backward) {
            topo[i]->_backward(topo[i]);
        }
    }

    free(topo);
}

// free memory
void free_value(Value *v) {
    free(v->_prev);
    free(v->_op);
    free(v);
}