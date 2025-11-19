#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>

typedef struct {
    uint8_t *x;
    uint8_t *y;
    uint32_t *out;
    uint32_t out_count;
} node_t;

node_t *init_graph(uint32_t *num_nodes_pointer, const char *x_file, const char *y_file, uint32_t x_length, uint32_t y_length) {
    uint32_t num_nodes = *num_nodes_pointer;
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
    *num_nodes_pointer = num_nodes;
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

typedef struct {
    uint32_t to;
    uint32_t from;
    uint8_t *adat;
    uint8_t *bdat;
} edge_t;

edge_t *calculate_edges(node_t *graph, uint32_t node_count, uint32_t dimension, uint64_t edge_count) {
    
}

