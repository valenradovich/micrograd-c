#ifndef NN_H
#define NN_H
#include "engine.h"

typedef struct Module {
    // all this are function pointers--
    // returns an array of Value pointers, representing the params
    Value **(*parameters)(struct Module*);
    // returns the number of params
    int (*parameters_count)(struct Module*);
    // sets the gradients of all params to zero
    void (*zero_grad)(struct Module*);
} Module;

typedef struct Neuron {
    Module base;
    Value **w;
    Value *b;
    int n_in;
    int nonlin;
} Neuron;

typedef struct Layer {
    Module base;
    Neuron **neurons;
    int n_in;
    int n_out;
} Layer;

typedef struct MLP {
    Module base;
    Layer **layers;
    int nlayers;
} MLP;

void module_zero_grad(Module *m);

// neuron's methods
Neuron *init_neuron(int n_in, int nonlin);
Value *neuron_call(Neuron *n, Value **x);
Value **neuron_parameters(Module *m);
int neuron_parameters_count(Module *m);
char *neuron_repr(Neuron *n);

// layer's methods
Layer *init_layer(int n_in, int n_out, int nonlin);
Value **layer_call(Layer *l, Value **x);
Value **layer_parameters(Module *m);
int layer_parameters_count(Module *m);
char *layer_repr(Layer *l);

// MLP's methods
MLP *init_mlp(int n_in, int *n_outs, int n_layers);
Value *mlp_call(MLP *mlp, Value **x);
Value **mlp_parameters(Module *m);
int mlp_parameters_count(Module *m);
char *mlp_repr(MLP *mlp);

// free memory
void free_neuron(Neuron *n);
void free_layer(Layer *l);
void free_mlp(MLP *mlp);

#endif // NN_H