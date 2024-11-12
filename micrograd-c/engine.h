#ifndef ENGINE_H
#define ENGINE_H

typedef struct Value {
    double data;
    double grad;
    void (*_backward)(struct Value *v);
    struct Value **_prev;
    int _prev_count;
    char *_op;
} Value;

// core methods
Value *init_value(double data, Value **children, int num_children, const char *op);
Value *add(Value *a, Value *b);
void add_backward(Value *v);
Value *mul(Value *a, Value *b);
void mul_backward(Value *v);
Value *power(Value *a, double power);
void power_backward(Value *v);
Value *relu(Value *a);
void relu_backward(Value *v);
Value *neg(Value *a);
Value *sub(Value *a, Value *b);
Value *division(Value *a, Value *b);
void build_topo(Value *v, Value ***sorted, int *size, int *capacity);
void backward(Value *v);
void free_value(Value *v);

#endif // ENGINE_H