#ifndef ENGINE_H
#define ENGINE_H

// structure definition
typedef struct Value {
    float data;
    float grad;
    void (*_backward)(struct Value*, struct Value*, struct Value*);
    struct Value *_prev[2];
    char *_op;
} Value;

// value initialization and memory management
Value *init_value(float data);
void value_free(Value *v);

// core operations
Value *add(Value *a, Value *b);
Value *mul(Value *a, Value *b);
Value *power(Value *a, Value *b);
Value *neg(Value *a);
Value *sub(Value *a, Value *b);
Value *divide(Value *a, Value *b);
void backward(Value *v);

// activation functions
Value *relu(Value *a);
Value *tanh_act(Value *a);

// utility functions
void value_print(Value *v);

#endif // ENGINE_H