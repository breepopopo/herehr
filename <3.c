#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>

typedef struct {
    uint8_t *x;
    uint8_t *y;
    uint32_t x_length;
    uint32_t y_length;
    uint32_t *out;
    uint32_t out_count;
} node_t;

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
    }
    fclose(fx);
    fclose(fy);

    #pragma omp parallel for
    for (uint32_t i = 0; i < num_nodes; ++i) {
        nodes[i].x_length = x_length;
        nodes[i].y_length = y_length;
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

//allocate cycle_lenghts first (lazy implementation)
uint32_t **find_cycles(node_t *nodes, uint32_t start_index, uint32_t end_index, uint32_t num_nodes, uint32_t *cycle_lenghts, uint32_t cycle_target) {
    uint32_t **cycles = malloc(sizeof(void *) * cycle_target), cycle_count = 0, thread_num = omp_get_max_threads();
    uint32_t init_size = nodes[start_index].out_count;
    #pragma omp parallel
    {
        uint32_t tid = omp_get_thread_num();
        uint32_t base = init_size / thread_num;
        uint32_t rem = init_size % thread_num;
        uint32_t chunk = base + (tid < rem ? 1 : 0);
        uint32_t start = tid * base + (tid < rem ? tid : rem);
        uint32_t end = start + chunk;
        uint32_t **local_paths = malloc(sizeof(void *) * (end - start));
        uint32_t local_path_lenght = 2;
        uint32_t local_paths_count = 0;
        for (uint32_t i = start; i < end; ++i) {
            if (nodes[start_index].out[i] == end_index) continue;
            local_paths[local_paths_count] = malloc(sizeof(uint32_t) * 2);
            local_paths[local_paths_count][0] = start_index;
            local_paths[local_paths_count++][1] = nodes[start_index].out[i];
        }
        while (cycle_count < cycle_target) {
            uint32_t **next_paths, next_paths_count = 0;
            uint32_t temp = 0;
            for (uint32_t i = 0; i < local_paths_count; ++i) {
                temp += nodes[local_paths[i][local_path_lenght - 1]].out_count;
            }
            next_paths = malloc(sizeof(void *) * temp);
            bool to_break = false;
            for (uint32_t i = 0; i < local_paths_count; ++i) {
                for (uint32_t j = 0; j < nodes[local_paths[i][local_path_lenght - 1]].out_count; ++j) {
                    uint32_t next_node = nodes[local_paths[i][local_path_lenght - 1]].out[j];
                    if (next_node == start_index) continue;
                    if (next_node == end_index) {
                        uint32_t the_cycle_count;
                        #pragma omp atomic capture
                        the_cycle_count = cycle_count++;
                        if (the_cycle_count < cycle_target) {
                            cycles[the_cycle_count] = malloc(sizeof(uint32_t) * (local_path_lenght + 1));
                            memcpy(cycles[the_cycle_count], local_paths[i], sizeof(uint32_t) * local_path_lenght);
                            cycles[the_cycle_count][local_path_lenght] = end_index;
                            cycle_lenghts[the_cycle_count] = local_path_lenght + 1;
                        } else {
                            to_break = true;
                            break;
                        }
                    } else {
                        bool unique = true;
                        for (uint32_t k = 0; k < local_path_lenght - 1; ++k) {
                            if (local_paths[i][k] == next_node) {
                                unique = false;
                                break;
                            }
                        }
                        if (!unique) {
                            continue;
                        } else {
                            next_paths[next_paths_count] = malloc(sizeof(uint32_t) * (local_path_lenght + 1));
                            memcpy(next_paths[next_paths_count], local_paths[i], sizeof(uint32_t) * local_path_lenght);
                            next_paths[next_paths_count++][local_path_lenght] = next_node;
                        }
                    }
                }
                if (to_break) break;
            }
            for (uint32_t i = 0; i < local_paths_count; ++i)
                free(local_paths[i]);
            free(local_paths);
            local_paths = next_paths;
            local_paths_count = next_paths_count;
            local_path_lenght++;
        }
        for (uint32_t i = 0; i < local_paths_count; ++i) {
            free(local_paths[i]);
        }
        free(local_paths);
    }
    return cycles;
}

