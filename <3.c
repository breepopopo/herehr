#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

typedef struct node node_t;

struct node {
    uint8_t *x;
    uint8_t *y;
    uint32_t x_length;
    uint32_t y_length;
    uint32_t *out;
    uint32_t out_count;
};

node_t *init_graph(uint32_t num_nodes, const char *x_file, const char *y_file, uint32_t x_length, uint32_t y_length) {
    FILE *fx = fopen(x_file, "rb");
    FILE *fy = fopen(y_file, "rb");
    if (!fx || !fy) {
        perror("Failed to open input files");
        if (fx) fclose(fx);
        if (fy) fclose(fy);
        return NULL;
    }

    node_t *nodes = malloc(sizeof(node_t) * num_nodes);
    if (!nodes) {
        perror("Failed to allocate memory for nodes");
        fclose(fx);
        fclose(fy);
        return NULL;
    }

    for (uint32_t i = 0; i < num_nodes; ++i) {
        nodes[i].x = malloc(x_length);
        nodes[i].y = malloc(y_length);
        if (!nodes[i].x || !nodes[i].y) {
            perror("Failed to allocate memory for node data");
            for (uint32_t j = 0; j <= i; ++j) {
                free(nodes[j].x);
                free(nodes[j].y);
            }
            free(nodes);
            fclose(fx);
            fclose(fy);
            return NULL;
        }

        size_t rx = fread(nodes[i].x, 1, x_length, fx);
        size_t ry = fread(nodes[i].y, 1, y_length, fy);
        if (rx != x_length || ry != y_length) {
            fprintf(stderr, "Error: unexpected EOF at node %u\n", i);
            num_nodes = i; // adjust for partial load
            break;
        }

        nodes[i].x_length = x_length;
        nodes[i].y_length = y_length;
    }
    fclose(fx);
    fclose(fy);

    #pragma omp parallel for
    for (uint32_t i = 0; i < num_nodes; ++i) {
        nodes[i].out = malloc(sizeof(uint32_t) * (num_nodes - 1));
        nodes[i].out_count = 0;
        for (uint32_t j = 0; j < i; ++j) {
            nodes[i].out[nodes[i].out_count++] = j;
        }
        for (uint32_t j = i + 1; j < num_nodes; ++j) {
            nodes[i].out[nodes[i].out_count++] = j;
        }
    }

    return nodes;
}

void free_graph(node_t *nodes, uint32_t num_nodes) {
    for (uint32_t i = 0; i < num_nodes; ++i) {
        free(nodes[i].x);
        free(nodes[i].y);
        free(nodes[i].out);
    }
    free(nodes);
}

//allocate cycle_lenghts first
uint32_t **find_cycles(node_t *nodes, uint32_t start_index, uint32_t end_index, uint32_t num_nodes, uint32_t *cycle_lenghts, uint32_t cycle_count) {
    uint32_t **cycles = malloc(sizeof(void *) * cycle_count);
    if (!cycles) {
        perror("Failed to allocate memory for cycles");
        return NULL;
    }
    

}