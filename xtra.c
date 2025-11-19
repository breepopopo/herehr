uint32_t **find_cycles(node_t *nodes, uint32_t start_index, uint32_t end_index, uint32_t num_nodes, uint32_t *cycle_lenghts, uint32_t cycle_target) {
    uint32_t **cycles = malloc(sizeof(void *) * cycle_target), cycle_count = 0, thread_num = omp_get_max_threads();
    uint32_t init_size = nodes[start_index].out_count;
    if (!cycles) {
        perror("Failed to allocate memory for cycles");
        return NULL;
    }

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
        if (!local_paths) {
            perror("Failed to allocate memory for local paths");
            return;
        }
        
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
            if (!next_paths) {
                perror("Failed to allocate memory for next paths");
                for (uint32_t i = 0; i < local_paths_count; ++i)
                    free(local_paths[i]);
                free(local_paths);
                return;
            }

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
                            if (!cycles[the_cycle_count]) {
                                perror("Failed to allocate memory for cycle");
                                to_break = true;
                                break;
                            }

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
                            if (!next_paths[next_paths_count]) {
                                perror("Failed to allocate memory for next path");
                                to_break = true;
                                break;
                            }

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