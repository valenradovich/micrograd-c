#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "engine.h"

// structures 
typedef struct Module {
    // all this are function pointers--
    // returns an array of Value pointers, representing the params
    Value** (*params)(struct Module*);
    
    // returns an integer representing the count of parameters 
    int (*params_count)(struct Module*);
    
    // resets the gradients of the parameters 
    void (*zero_grad)(struct Module*);
} Module;

typedef struct Neuron {
    int n_in;
    Value **w;
    Value *b;
    int nonlin;
}Neuron;

typedef struct Layer {
    Neuron **neurons;
    int n_in;
    int n_out;
}Layer;

typedef struct MLP {
    Layer **layers;
    int n_layers;
}MLP;


// constructors
Neuron *init_neuron(int n_in, int nonlin) {
    Neuron *n = malloc(sizeof(Neuron));
    n->n_in = n_in;
    n->nonlin = nonlin;
    
    n->w = malloc(sizeof(Value*) * n_in);
    for(int i = 0; i < n_in; i++) {
        n->w[i] = value_create((float)rand() / RAND_MAX * 2 - 1);
    }
    n->b = value_create(0.0);
    return n;
}

Layer *init_layer(int n_in, int n_out, int nonlin) {
    Layer *l = malloc(sizeof(Layer));
    l->neurons = malloc(sizeof(Neuron*) * n_out);
    for (int i = 0; i < n_out; i++) {
        l->neurons[i] = init_neuron(n_in, nonlin);
    }
    l->n_in = n_in;
    l->n_out = n_out;
    return l;
}

MLP *init_mlp(int n_in, int *n_out, int n_layers) {
    MLP *mlp = malloc(sizeof(MLP));
    mlp->layers = malloc(sizeof(Layer*) * n_layers);
    mlp->n_layers = n_layers;

    for (int i = 0; i < n_layers; i++) {
        mlp->layers[i] = init_layer(i == 0 ? n_in : n_out[i-1], n_out[i], 1);
    }
    return mlp;
}

// call operations
Value *neuron_call(Neuron *n, Value **x) {
    Value *w_sum= n->b;
    for (int i = 0; i < n->n_in; i++) {
        // calculating wi * xi + b
        w_sum= add(w_sum, mul(n->w[i], x[i]));
    }
    Value *out = tanh_act(w_sum);
    return out;
}

Value **layer_call(Layer *l, Value **x) {
    Value **out = malloc(l->n_out * sizeof(Value *));
    for (int i = 0; i < l->n_out; i++) {
        out[i] = neuron_call(l->neurons[i], x);
    }
    return out;
}

Value *mlp_call(MLP *mlp, Value **x)  {
    Value **out = x; 
    for (int i = 0; i < mlp->n_layers; i++) {
        out = layer_call(mlp->layers[i], out);
    }
    return out[0];
}






