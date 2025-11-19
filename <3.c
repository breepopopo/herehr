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

uint32_t *unlimited_power(node_t *graph, uint32_t node_count, uint32_t start_node, uint32_t end_node, uint32_t cycle_lenght) {
    uint32_t thread_num = omp_get_max_threads();
    uint32_t **starting_range = malloc(sizeof(void *) * thread_num);
    uint32_t *starting_lenghts;
    
}

uint32_t *unlimited_power(node_t *graph, uint32_t node_count, uint32_t start_node, uint32_t end_node, uint32_t cycle_length) {
    uint32_t thread_num = omp_get_max_threads();
    uint32_t **starting_paths = malloc(sizeof(void *) * thread_num);
    uint32_t *starting_path_len = malloc(sizeof(uint32_t) * thread_num);
    uint32_t path_num = 0;
    if (i < graph[start_node].out_count) {
        starting_paths[i] = malloc(sizeof(uint32_t));
        starting_paths[i][0] = start_node;
        starting_path_len[i] = 1;
        ++path_num;
    } else {
        
    }
    while (path_num < thread_num) {
        starting_paths[path_num] = NULL;
        starting_path_len[path_num] = 0;
        path_num++;
    }
    uint32_t *result = NULL;
    int found = 0;
    #pragma omp parallel shared(found, result)
    {
        int tid = omp_get_thread_num();
        if (found) return;
        uint32_t *seed = starting_paths[tid];
        uint32_t seed_len = starting_path_len[tid];
        uint32_t *cycle = malloc(sizeof(uint32_t) * cycle_length);
        uint32_t *max_counter = malloc(sizeof(uint32_t) * cycle_length);
        uint32_t *counter = calloc(cycle_length, sizeof(uint32_t));
        if (!cycle || !counter || !max_counter) goto cleanup;
        for (uint32_t i = 0; i < seed_len; i++) cycle[i] = seed[i];
        uint32_t depth = seed_len - 1;
        for (uint32_t i = 0; i < seed_len; i++) {
            uint32_t node = cycle[i];
            max_counter[i] = graph[node].out_count;
            counter[i] = 0;
        }
        while (!found) {
            if (depth + 1 == cycle_length) {
                if (cycle[depth] == end_node) {
                    #pragma omp critical
                    {
                        if (!found) {
                            found = 1;
                            result = malloc(sizeof(uint32_t) * cycle_length);
                            memcpy(result, cycle, sizeof(uint32_t) * cycle_length);
                        }
                    }
                }
                if (depth == seed_len - 1)
                    break;
                depth--;
                continue;
            }
            if (counter[depth] == max_counter[depth]) {
                if (depth == seed_len - 1)
                    break;
                depth--;
                continue;
            }
            uint32_t next = graph[cycle[depth]].out[counter[depth]];
            counter[depth]++;
            depth++;
            cycle[depth] = next;
            max_counter[depth] = graph[next].out_count;
            counter[depth] = 0;
        }
    cleanup:
        free(cycle);
        free(counter);
        free(max_counter);
    }
    for (uint32_t i = 0; i < thread_num; i++) free(starting_paths[i]);
    free(starting_paths);
    free(starting_path_len);
    return result;
}

edge_t *calculate_edges(node_t *graph, uint32_t node_count, uint32_t dimension, uint64_t edge_count) {

}

//A_1 x_11 = x_21




//i mean the core is simple graph exploration, start at a cycle,