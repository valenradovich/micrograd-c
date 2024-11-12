#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "nn.h"
#include "engine.h"

// sets gradients of all parameters to zero
void module_zero_grad(Module *m) {
    Value **params = m->parameters(m);
    int count = m->parameters_count(m);
    for (int i = 0; i < count; i++) {
        params[i]->grad = 0;
    }
    free(params);
}

// 
Neuron *init_neuron(int n_in, int nonlin) {
    Neuron *n = malloc(sizeof(Neuron));
    n->w = malloc(n_in * sizeof(Value*));
    for (int i = 0; i < n_in; i++) {
        // initialize weights with random values between -1 and 1
        n->w[i] = init_value((double)rand() / RAND_MAX * 2 - 1, NULL, 0, "");
    }
    n->b = init_value(0, NULL, 0, "");  // initialize bias to 0
    n->n_in = n_in;
    n->nonlin = nonlin;
    n->base.parameters = neuron_parameters;
    n->base.parameters_count = neuron_parameters_count;
    n->base.zero_grad = module_zero_grad;
    return n;
}

Value *neuron_call(Neuron *n, Value **x) {
    Value *act = n->b;  
    for (int i = 0; i < n->n_in; i++) {
        // wi * xi + b
        act = add(act, mul(n->w[i], x[i]));
    }
    return n->nonlin ? relu(act) : act;  // apply ReLU if nonlinear
}

Value **neuron_parameters(Module* m) {
    Neuron *n = (Neuron*)m;
    Value **params = malloc((n->n_in + 1) * sizeof(Value*));
    memcpy(params, n->w, n->n_in * sizeof(Value*));  // copy weights
    params[n->n_in] = n->b;  // add bias as the last parameter
    return params;
}

int neuron_parameters_count(Module* m) {
    Neuron *n = (Neuron*)m;
    return n->n_in + 1;  // number of inputs + bias
}

char *neuron_repr(Neuron *n) {
    char *repr = malloc(50 * sizeof(char));
    snprintf(repr, 50, "%sNeuron(%d)", n->nonlin ? "ReLU" : "Linear", n->n_in);
    return repr;
}

// layer's methods
Layer *init_layer(int n_in, int n_out, int nonlin) {
    Layer *l = malloc(sizeof(Layer));
    l->neurons = malloc(n_out * sizeof(Neuron*));
    for (int i = 0; i < n_out; i++) {
        l->neurons[i] = init_neuron(n_in, nonlin);
    }
    l->n_in = n_in;
    l->n_out = n_out;
    l->base.parameters = layer_parameters;
    l->base.parameters_count = layer_parameters_count;
    l->base.zero_grad = module_zero_grad;
    return l;
}

Value **layer_call(Layer *l, Value **x) {
    Value **out = malloc(l->n_out * sizeof(Value*));
    for (int i = 0; i < l->n_out; i++) {
        out[i] = neuron_call(l->neurons[i], x);
    }
    return out;
}

Value **layer_parameters(Module *m) {
    Layer *l = (Layer*)m;
    int count = layer_parameters_count(m);
    Value **params = malloc(count * sizeof(Value*));
    int index = 0;
    for (int i = 0; i < l->n_out; i++) {
        Value **neuron_params = neuron_parameters((Module*)l->neurons[i]);
        int neuron_param_count = neuron_parameters_count((Module*)l->neurons[i]);
        // passing in the parameters of each neuron into the layer's parameter array
        memcpy(params + index, neuron_params, neuron_param_count * sizeof(Value*));
        index += neuron_param_count;
        free(neuron_params);
    }
    return params;
}

int layer_parameters_count(Module *m) {
    Layer *l = (Layer*)m;
    int count = 0;
    for (int i = 0; i < l->n_out; i++) {
        count += neuron_parameters_count((Module*)l->neurons[i]);
    }
    return count;
}

char *layer_repr(Layer *l) {
    char *repr = malloc(200 * sizeof(char));
    char *temp = malloc(50 * sizeof(char));
    strcpy(repr, "Layer of [");
    for (int i = 0; i < l->n_out; i++) {
        char *neuron_str = neuron_repr(l->neurons[i]);
        strcat(repr, neuron_str);
        if (i < l->n_out - 1) strcat(repr, ", ");
        free(neuron_str);
    }
    strcat(repr, "]");
    free(temp);
    return repr;
}

// MLP's methods
MLP *init_mlp(int n_in, int *n_outs, int n_layers) {
    MLP *mlp = malloc(sizeof(MLP));
    mlp->layers = malloc(n_layers * sizeof(Layer*));
    mlp->nlayers = n_layers;
    
    int prev_n_out = n_in;
    for (int i = 0; i < n_layers; i++) {
        // all layers but the last are non-linear
        mlp->layers[i] = init_layer(prev_n_out, n_outs[i], i != n_layers - 1);
        prev_n_out = n_outs[i];
    }
    
    mlp->base.parameters = mlp_parameters;
    mlp->base.parameters_count = mlp_parameters_count;
    mlp->base.zero_grad = module_zero_grad;
    return mlp;
}

Value *mlp_call(MLP *mlp, Value **x) {
    Value **out = x;
    for (int i = 0; i < mlp->nlayers; i++) {
        out = layer_call(mlp->layers[i], out);
    }
    return out[0];  
}

Value **mlp_parameters(Module *mod) {
    MLP *mlp = (MLP*)mod;
    int count = mlp_parameters_count(mod);
    Value **params = malloc(count * sizeof(Value*));
    int index = 0;
    for (int i = 0; i < mlp->nlayers; i++) {
        Value **layer_params = layer_parameters((Module*)mlp->layers[i]);
        int layer_param_count = layer_parameters_count((Module*)mlp->layers[i]);
        // passing in the parameters of each layer into the MLP's parameter array
        memcpy(params + index, layer_params, layer_param_count * sizeof(Value*));
        index += layer_param_count;
        free(layer_params);
    }
    return params;
}

int mlp_parameters_count(Module *mod) {
    MLP *mlp = (MLP*)mod;
    int count = 0;
    for (int i = 0; i < mlp->nlayers; i++) {
        count += layer_parameters_count((Module*)mlp->layers[i]);
    }
    return count;
}

char *mlp_repr(MLP *mlp) {
    char *repr = malloc(500 * sizeof(char));
    strcpy(repr, "MLP of [");
    for (int i = 0; i < mlp->nlayers; i++) {
        char* layer_str = layer_repr(mlp->layers[i]);
        strcat(repr, layer_str);
        if (i < mlp->nlayers - 1) strcat(repr, ", ");
        free(layer_str);
    }
    strcat(repr, "]");
    return repr;
}

// free memory
void free_neuron(Neuron *n) {
    for (int i = 0; i < n->n_in; i++) {
        free_value(n->w[i]);
    }
    free(n->w);
    free_value(n->b);
    free(n);
}

void free_layer(Layer *l) {
    for (int i = 0; i < l->n_out; i++) {
        free_neuron(l->neurons[i]);
    }
    free(l->neurons);
    free(l);
}

void free_mlp(MLP *mlp) {
    for (int i = 0; i < mlp->nlayers; i++) {
        free_layer(mlp->layers[i]);
    }
    free(mlp->layers);
    free(mlp);
}