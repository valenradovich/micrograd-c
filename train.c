#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "micrograd-c/nn.h"
#include "micrograd-c/engine.h"
#define SIZE 200

void load_dataset(const char *filename, double X[SIZE][2], int y[SIZE]) {
    FILE *f = fopen(filename, "r");
    if (f == NULL) {
        printf("Error opening file: %s\n", filename);
        exit(1);
    }

    char line[1024];
    fgets(line, sizeof(line), f);  // skipping header

    for (int i = 0; i < SIZE; i++) {
        if (fscanf(f, "%lf,%lf,%d", &X[i][0], &X[i][1], &y[i]) != 3) {
            printf("Error reading line %d\n", i + 1);
            exit(1);
        }
    }
    fclose(f);
}

double margin_loss(MLP *model, double X[SIZE][2], int y[SIZE], double *accuracy) {
    double total_loss = 0.0;
    int correct = 0;
    
    model->base.zero_grad((Module*)model);

    for (int i = 0; i < SIZE; i++) {
        // forward pass
        Value* inputs[2];
        for (int j = 0; j < 2; j++) {
            inputs[j] = init_value(X[i][j], NULL, 0, "");
        }
        Value* output = mlp_call(model, inputs);
        
        // compute loss
        Value* target = init_value(y[i] * 2.0 - 1.0, NULL, 0, "");  // converting 0/1 to -1/+1
        Value* margin_loss = relu(add(neg(mul(target, output)), init_value(1.0, NULL, 0, "")));
        total_loss += margin_loss->data;

        // accumulate gradients
        backward(margin_loss);

        // compute accuracy
        if ((y[i] == 1 && output->data > 0) || (y[i] == 0 && output->data <= 0)) {
            correct++;
        }

        for (int j = 0; j < 2; j++) {
            free_value(inputs[j]);
        }
        free_value(output);
        free_value(target);
        free_value(margin_loss);
    }

    // regularization loss -- L2 regularization
    double reg_loss = 0.0;
    double alpha = 1e-4;
    Value** params = model->base.parameters((Module*)model);
    int param_count = model->base.parameters_count((Module*)model);
    for (int p = 0; p < param_count; p++) {
        reg_loss += params[p]->data * params[p]->data;
    }
    reg_loss *= alpha;
    total_loss += reg_loss;
    total_loss /= SIZE;

    *accuracy = (double)correct / SIZE * 100.0;

    free(params);

    return total_loss;
}

void train(MLP *model, double X[SIZE][2], int y[SIZE]) {
    for (int epoch = 0; epoch < 100; epoch++) {
        double accuracy = 0.0;
        double avg_loss = margin_loss(model, X, y, &accuracy);

        // updating weights
        double learning_rate = 0.01; //1.0 - (0.9 * epoch / 100.0) -- learning rate decay is not working well
        Value **params = model->base.parameters((Module*)model);
        int param_count = model->base.parameters_count((Module*)model);
        for (int i = 0; i < param_count; i++) {
            params[i]->data -= learning_rate * params[i]->grad;
        }
        free(params);

        printf("step %d loss %f, accuracy %f%%\n", epoch, avg_loss, accuracy);

        // zero out gradients for next epoch
        model->base.zero_grad((Module*)model);
    }
}

int main(void) {
    srand(time(NULL));

    // load data
    double X[SIZE][2];
    int y[SIZE];
    load_dataset("data/moons_dataset.csv", X, y);

    // model
    int layer_sizes[] = {2, 16, 16, 1};
    MLP* model = init_mlp(2, layer_sizes, 3);

    // training loop
    train(model, X, y);

    free_mlp(model);

    return 0;
}