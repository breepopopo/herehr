
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <omp.h>
#include <immintrin.h>

typedef struct node node_t;
typedef struct edge edge_t;
typedef struct graph graph_t;
typedef struct num_matrix num_matrix_t;
typedef struct sym_matrix sym_matrix_t;
typedef struct equation equation_t;
typedef struct multinomial multinomial_t;
typedef struct path path_t;

struct graph {
  node_t *nodes;
  edge_t *edges;
  size_t node_count;
  size_t edge_count;
};

struct edge {
  bool type;
  size_t from;
  size_t to;
  num_matrix_t *A_num;
  num_matrix_t *B_num;
  sym_matrix_t *A_sym;
  sym_matrix_t *B_sym;
};

struct node {
  bool visited;
  bool *x;
  bool *y;
  size_t x_count;
  size_t y_count;
  size_t *out;
  size_t out_count;
};

struct num_matrix {
  bool *data;
  size_t size;
};

struct sym_matrix {
  equation_t *equations;
  size_t equ_count;
};

struct equation {
  multinomial_t *multinomials;
  size_t mon_count;
};

struct multinomial {
  size_t *variables;
  size_t var_count;
};

struct path {
  size_t *edges;
  size_t path_lenght;
};

size_t variable_generator = 0;

graph_t *init_graph(uint8_t *x_in, uint8_t *y_in, size_t node_count, size_t x_size, size_t y_size) {
  graph_t *g = malloc(sizeof(graph_t));
  g->node_count = node_count;
  g->edge_count = node_count * (node_count - 1);
  g->nodes = malloc(sizeof(node_t) * node_count);
  g->edges = malloc(sizeof(edge_t) * g->edge_count);

  #pragma omp parallel for
  for (size_t i = 0; i < node_count; ++i) {
    g->nodes[i].visited = 0;
    g->nodes[i].x_count = x_size;
    g->nodes[i].y_count = y_size;
    g->nodes[i].x = malloc(sizeof(bool) * x_size);
    g->nodes[i].y = malloc(sizeof(bool) * y_size);
    for (size_t j = 0; j < x_size; ++j) {
      g->nodes[i].x[j] = (x_in[i * x_size + j] != 0);
    }
    for (size_t j = 0; j < y_size; ++j) {
      g->nodes[i].y[j] = (y_in[i * y_size + j] != 0);
    }
    g->nodes[i].out_count = node_count - 1;
    g->nodes[i].out = malloc(sizeof(size_t) * (node_count - 1));
    for (size_t j = 0; j < node_count; ++j) {
      if (i != j) {
        size_t edge_index = i * (node_count - 1) + (j < i ? j : j - 1);
        g->nodes[i].out[edge_index] = edge_index;
        g->edges[edge_index].type = 0;
        g->edges[edge_index].from = i;
        g->edges[edge_index].to = j;
        g->edges[edge_index].A_num = NULL;
        g->edges[edge_index].B_num = NULL;
        g->edges[edge_index].A_sym = malloc(sizeof(sym_matrix_t));
        g->edges[edge_index].A_sym->equ_count = x_size;
        g->edges[edge_index].A_sym->equations = malloc(sizeof(equation_t) * x_size);
        for (size_t k = 0; k < x_size; ++k) {
          g->edges[edge_index].A_sym->equations[k].mon_count = 1;
          g->edges[edge_index].A_sym->equations[k].multinomials = malloc(sizeof(multinomial_t));
          g->edges[edge_index].A_sym->equations[k].multinomials[0].var_count = 1;
          g->edges[edge_index].A_sym->equations[k].multinomials[0].variables = malloc(sizeof(size_t));
          #pragma omp atomic
          g->edges[edge_index].A_sym->equations[k].multinomials[0].variables[0] = variable_generator++;
        }
        g->edges[edge_index].B_sym = malloc(sizeof(sym_matrix_t));
        g->edges[edge_index].B_sym->equ_count = y_size;
        g->edges[edge_index].B_sym->equations = malloc(sizeof(equation_t) * y_size);
        for (size_t k = 0; k < y_size; ++k) {
          g->edges[edge_index].B_sym->equations[k].mon_count = 1;
          g->edges[edge_index].B_sym->equations[k].multinomials = malloc(sizeof(multinomial_t));
          g->edges[edge_index].B_sym->equations[k].multinomials[0].var_count = 1;
          g->edges[edge_index].B_sym->equations[k].multinomials[0].variables = malloc(sizeof(size_t));
          #pragma omp atomic
          g->edges[edge_index].B_sym->equations[k].multinomials[0].variables[0] = variable_generator++;
        }
      }
    }
  }
  return g;
}

void free_graph(graph_t *g) {
  for (size_t i = 0; i < g->node_count; ++i) {
    free(g->nodes[i].x);
    free(g->nodes[i].y);
  }
  free(g->nodes);

  for (size_t i = 0; i < g->edge_count; ++i) {
    if (g->edges[i].A_num) {
      free(g->edges[i].A_num->data);
      free(g->edges[i].A_num);
    }
    if (g->edges[i].B_num) {
      free(g->edges[i].B_num->data);
      free(g->edges[i].B_num);
    }
    if (g->edges[i].A_sym) {
      for (size_t j = 0; j < g->edges[i].A_sym->equ_count; ++j) {
        for (size_t k = 0; k < g->edges[i].A_sym->equations[j].mon_count; ++k) {
          free(g->edges[i].A_sym->equations[j].multinomials[k].variables);
        }
        free(g->edges[i].A_sym->equations[j].multinomials);
      }
      free(g->edges[i].A_sym->equations);
      free(g->edges[i].A_sym);
    }
    if (g->edges[i].B_sym) {
      for (size_t j = 0; j < g->edges[i].B_sym->equ_count; ++j) {
        for (size_t k = 0; k < g->edges[i].B_sym->equations[j].mon_count; ++k) {
          free(g->edges[i].B_sym->equations[j].multinomials[k].variables);
        }
        free(g->edges[i].B_sym->equations[j].multinomials);
      }
      free(g->edges[i].B_sym->equations);
      free(g->edges[i].B_sym);
    }
  }
  free(g->edges);
  free(g);
}

typedef struct {
  size_t *data;
  size_t head;
  size_t tail;
  size_t capacity;
} queue_t;

void enqueue(queue_t *q, size_t value) {
  if (q->tail >= q->capacity) {
    q->capacity *= 2;
    q->data = realloc(q->data, sizeof(size_t) * q->capacity);
  }
  q->data[q->tail++] = value;
}

path_t *bfs_find_minimal_cycles_including_edge(graph_t *g, size_t start_edge_index, size_t *cycle_count) {
  
  g->nodes[g->edges[start_edge_index].from].visited = 1;
  g->nodes[g->edges[start_edge_index].to].visited = 1;
  
  size_t *next_frontier_sizes = malloc(sizeof(size_t) * omp_get_num_threads());
  size_t frontier_size = g->nodes[g->edges[start_edge_index].to].out_count
  size_t *next_frontier, *frontier = malloc(sizeof(size_t) * frontier_size);
  #pragma omp parallel for
  for (size_t i = 0; i < frontier_size; ++i) {
    frontier[i] = g->nodes[g->edges[start_edge_index].to].out[i];
  }
  size_t total = 0;
  #pragma omp parallel for
  for (size_t i = 0; i < g->nodes[g->edges[start_edge_index].to].out_count; ++i) {
	 next_frontier_sizes[i] = g->nodes[g->nodes[g->edges[start_edge_index].to].out[i];
	 #pragma atomic update
	 total += g->nodes[g->nodes[g->edges[start_edge_index].to].out[i]].out_count;
  }

	next_frontier = malloc(sizeof(size_t) * total);
	
  #pragma omp parallel
  {
    size_t thread_num = omp_get_thread_num();
    size_t num_threads = omp_get_num_threads();
    size_t chunk_size = (frontier_size + num_threads - 1) / num_threads;
    size_t local_start = thread_num * chunk_size;
    size_t local_end = (local_start + chunk_size > frontier_size) ? frontier_size : local_start + chunk_size;
	
	size_t local_mem_size = 0;
	for (size_t i = local_start; i < local_end; ++i) {
		local_mem_size += g->nodes[i].out_count;
	}
	next_frontier_sizes[thread_num] = local_mem_size;
	
	
    for (size_t i = local_start; i < local_end; ++i) {
      size_t edge_index = frontier[i];
      size_t to_node = g->edges[edge_index].to;
      #pragma omp atomic capture
      {
        bool was_visited = g->nodes[to_node].visited;
        g->nodes[to_node].visited = 1;
      }
      if (!was_visited) {
        // Add to next frontier
        

      }
    }
  }
  
}
