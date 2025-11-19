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

uint32_t **init_POWER(node_t *nodes, uint32_t start_index, uint32_t end_index, uint32_t num_nodes, uint32_t path_target, uint32_t *out_count) {
    uint32_t thread_num = omp_get_max_threads();
    uint32_t init_size  = nodes[start_index].out_count;
    uint32_t **global_paths = NULL;
    uint32_t global_count = 0;
    uint32_t global_cap = 0;
    omp_lock_t lock;
    omp_init_lock(&lock);

    #pragma omp parallel
    {
        uint32_t tid = omp_get_thread_num();
        uint32_t base = init_size / thread_num;
        uint32_t rem  = init_size % thread_num;
        uint32_t chunk = base + (tid < rem ? 1 : 0);
        uint32_t start = tid * base + (tid < rem ? tid : rem);
        uint32_t end   = start + chunk;
        uint32_t **local_paths = malloc(sizeof(uint32_t*) * chunk);
        if (!local_paths) {
            fprintf(stderr, "Thread %u: Failed to allocate local paths\n", tid);
            return;
        }

        uint32_t local_paths_count = 0;
        uint32_t local_length = 2;
        for (uint32_t i = start; i < end; ++i) {
            uint32_t next = nodes[start_index].out[i];
            if (next == end_index) continue;
            local_paths[local_paths_count] = malloc(sizeof(uint32_t) * 2);
            if (!local_paths[local_paths_count]) {
                fprintf(stderr, "Thread %u: Failed to allocate path\n", tid);
                continue;
            }

            local_paths[local_paths_count][0] = start_index;
            local_paths[local_paths_count++][1] = next;
        }

        while (true) {
            uint32_t global_copy;
            #pragma omp atomic read
            global_copy = global_count;
            if (global_copy >= path_target || local_paths_count == 0) break;
            uint32_t estimate = 0;
            for (uint32_t i = 0; i < local_paths_count; ++i) estimate += nodes[local_paths[i][local_length - 1]].out_count;
            uint32_t **next_paths = malloc(sizeof(uint32_t*) * estimate);
            uint32_t next_count = 0;
            for (uint32_t i = 0; i < local_paths_count; ++i) {
                uint32_t tail = local_paths[i][local_length - 1];
                uint32_t outc = nodes[tail].out_count;
                for (uint32_t j = 0; j < outc; ++j) {
                    uint32_t next = nodes[tail].out[j];
                    if (next == start_index) continue;
                    if (next == end_index) continue;
                    bool cycle = false;
                    for (uint32_t k = 0; k < local_length; ++k) {
                        if (local_paths[i][k] == next) {
                            cycle = true;
                            break;
                        }
                    }
                    if (cycle) continue;
                    uint32_t *np = malloc(sizeof(uint32_t) * (local_length + 1));
                    if (!np) {
                        fprintf(stderr, "Thread %u: Failed to allocate new path\n", tid);
                        continue;
                    }
                    
                    memcpy(np, local_paths[i], sizeof(uint32_t) * local_length);
                    np[local_length] = next;
                    next_paths[next_count++] = np;
                    #pragma omp atomic
                    ++global_count;
                }
                free(local_paths[i]);
            }
            free(local_paths);
            local_paths = next_paths;
            local_paths_count = next_count;
            local_length++;
        }
        for (uint32_t i = 0; i < local_paths_count; ++i) {
            if (nodes[local_paths[i][local_length - 1]].out_count == 0) {
                free(local_paths[i]);
            } else {
                omp_set_lock(&lock);
                if (global_count >= global_cap) {
                    global_cap = (global_cap == 0 ? 1024 : global_cap * 2);
                    global_paths = realloc(global_paths, sizeof(uint32_t*) * global_cap);
                }
                global_paths[global_count++] = local_paths[i];
                omp_unset_lock(&lock);
            }
        }
        free(local_paths);
    }
    omp_destroy_lock(&lock);
    *out_count = global_count;
    return global_paths;
}



uint32_t *unlimited_power(node_t *graph, uint32_t node_count, uint32_t start_node, uint32_t end_node, uint32_t cycle_lenght) {
    uint32_t thread_num = omp_get_max_threads();
    
    

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