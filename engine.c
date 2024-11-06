#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// defining struct 
typedef struct Value {
    float data;
    float grad;
    void (*_backward)(struct Value*, struct Value*, struct Value*); // function pointer
    struct Value *_prev[2]; 
    char *_op;
} Value;

// backward functions to each opeartion
static void add_backward(Value *out, Value *a, Value *b) {
    a->grad += out->grad;
    b->grad += out->grad;
}

static void mul_backward(Value *out, Value *a, Value *b) {
    a->grad += b->data * out->grad;
    b->grad += a->data * out->grad;
}

static void relu_backward(Value *out, Value *a, Value *b) {
    if (a->data > 0.0f) {
            a->grad += out->grad;
        }
}

static void tanh_backward(Value *out, Value *a, Value *b) {
    float tanh_squared = out->data * out->data;
    a->grad += (1 - tanh_squared) * out->grad;
}

// defining operations 
Value *value_create(float data) {
    Value *v = malloc(sizeof(Value));
    v->data = data;
    v->grad = 0.0;
    v->_backward = NULL;
    v->_prev[0] = v->_prev[1] = NULL; 
    v->_op = NULL;
    return v;
}

Value *add(Value *a, Value *b) {
    Value *out = value_create(a->data + b->data);
    out->_prev[0] = a;
    out->_prev[1] = b;
    out->_op = "+";
    out->_backward = add_backward;
    return out;
}

Value *mul(Value *a, Value *b) {
    Value *out = value_create(a->data * b->data);
    out->_prev[0] = a;
    out->_prev[1] = b;
    out->_op = "*";
    out->_backward = mul_backward;
    return out;
}

Value *relu(Value *a) {
    Value *out = value_create(a->data < 0 ? 0 : a->data);
    out->_prev[0] = a;
    out->_prev[1] = NULL;
    out->_op = "ReLU";
    out->_backward = relu_backward;
    return out;
}

Value *tanh_act(Value *a) {
    float x = a->data;
    float exp_x = exp(x);
    float exp_neg_x = exp(-x);
    float tanh_x = (exp_x - exp_neg_x) / (exp_x + exp_neg_x);
    Value *out = value_create(tanh_x);
    out->_prev[0] = a;
    out->_prev[1] = NULL;
    out->_op = "tanh";
    out->_backward = tanh_backward;
    return out;
}

void value_print(Value *v) {
    printf("Value(data=%.2f)\n",v->data);
}

// main backward function
void build_topo(Value *v, Value **topo, int *topo_size, Value **visited, int *visited_size) {
    // checking if node was visited
    for (int i = 0; i < *visited_size; i++) {
        if (visited[i] == v) return;
    }
    
    // adding to visited
    visited[*visited_size] = v;
    (*visited_size)++;
    
    // recursive on childrens 
    for (int i = 0; i < 2; i++) {
        if (v->_prev[i] != NULL) {
            build_topo(v->_prev[i], topo, topo_size, visited, visited_size);
        }
    }
    
    // adding to topological order
    topo[*topo_size] = v;
    (*topo_size)++;
}

// backpropagation
void backward(Value *v) {
    size_t max_size = sizeof(Value*) * 100;  
    
    Value **topo = malloc(max_size);
    Value **visited = malloc(max_size);
    
    if (!topo || !visited) {
        fprintf(stderr, "Memory allocation failed\n");
        free(topo);
        free(visited);
        return;
    }
    
    int topo_size = 0;
    int visited_size = 0;
    
    build_topo(v, topo, &topo_size, visited, &visited_size);

    //for (size_t i = 0; i < topo_size; i++)
    //{
        //printf("%f\n", topo[i]->data);
    //}
    
    v->grad = 1.0; // root gradient to 1

    // iterate through nodes in reverse topological order to start the backpropagation from output
    for (int i = topo_size - 1; i >= 0; i--) {
        Value *current = topo[i];
        // calling backward function with the node and its children
        if (current->_backward != NULL) {
            current->_backward(current, current->_prev[0], current->_prev[1]);
        }
    }
    
    free(topo);
    free(visited);
}

// value free function
void value_free(Value *v) {
    if (v != NULL) {
        free(v);
    }
}

int main() {
    // inputs x1,x2
    Value *x1 = value_create(2.0);
    Value *x2 = value_create(0.0);
    
    // weights w1,w2
    Value *w1 = value_create(-3.0);
    Value *w2 = value_create(1.0);
    
    // bias
    Value *b = value_create(6.88137);
    
    // forward pass: f(x) = relu(w1*x1 + w2*x2 + b)
    Value *x1w1 = mul(x1, w1);
    printf("x1w1: %.2f\n", x1w1->data);
    
    Value *x2w2 = mul(x2, w2);
    printf("x2w2: %.2f\n", x2w2->data);
    
    Value *x1w1x2w2 = add(x1w1, x2w2);
    printf("x1w1x2w2: %.2f\n", x1w1x2w2->data);
    
    Value *n = add(x1w1x2w2, b);
    printf("n: %.2f\n", n->data);
    
    //Value *o = relu(n);
    Value *o = tanh_act(n);
    
    printf("forward pass result: ");
    value_print(o);
    
    // backward pass 
    o->grad = 1.0;
    backward(o); 

    printf("\nGradients:\n");
    printf("x1.grad = %.2f\n", x1->grad);
    printf("x2.grad = %.2f\n", x2->grad);
    printf("w1.grad = %.2f\n", w1->grad);
    printf("w2.grad = %.2f\n", w2->grad);
    printf("b.grad = %.2f\n", b->grad);
    
    value_free(x1);
    value_free(x2);
    value_free(w1);
    value_free(w2);
    value_free(b);
    value_free(x1w1);
    value_free(x2w2);
    value_free(x1w1x2w2);
    value_free(n);
    value_free(o);
    
    return 0;
}




