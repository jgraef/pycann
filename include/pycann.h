/*
 pycann - Neural network library
 A Python/C hybrid for fast neural networks in Python
 Copyright (C) 2010  Janosch Gräf <janosch.graef@gmx.net>

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Lesser General Public License for more details.

 You should have received a copy of the GNU Lesser General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef _PYCANN_H_
#define _PYCANN_H_

// Define this to use multi-threading
//#define PYCANN_THREADING

#ifdef PYCANN_THREADING
#include <pthread.h>
#endif /* PYCANN_THREADING */


#define PYCANN_WEIGHT(net, a, b) ((net)->weights[(a)*(net)->size+(b)])

typedef float pycann_float_t;

typedef struct pycann_struct pycann_t;

#ifdef PYCANN_THREADING
typedef struct pycann_thread_struct pycann_thread_t;

struct pycann_thread_struct {
  pthread_t thread;
  unsigned int first_neuron;
  unsigned int last_neuron; // actually this is the neuron after the last one
  unsigned int steps;
};
#endif /* PYCANN_THREADING */

struct pycann_struct {
  // Number of neurons
  unsigned int size;

  // Network-wide learning rate
  pycann_float_t learning_rate;

  // Gamma coefficients for Hebbian plasticity rule
  pycann_float_t gamma[4];

  // Weights (see macro PYCANN_WEIGHT)
  pycann_float_t *weights;

  // Thresholds
  pycann_float_t *thresholds;

  // Activation potentials
  pycann_float_t *activations;

  // Modularity connections
  pycann_float_t *mod_weights;
  //unsigned int *mod_neurons;
  pycann_float_t **mod_neurons;

  // Memory usage
  unsigned int memory_usage;

  // Inputs
  unsigned int num_inputs;
  pycann_float_t *inputs;

  // Outputs
  unsigned int num_outputs;

#ifdef PYCANN_THREADING
  // Threading
  unsigned int num_threads;
  pycann_thread_t *threads;
#endif /* PYCANN_THREADING */
};

#define PYCANN_FILE_SIGNATURE "PYCANN_NETWORK\0\1"
#define PYCANN_FILE_SIGNATURE_LENGTH 16
struct pycann_file_header {
  char signature[PYCANN_FILE_SIGNATURE_LENGTH];

  unsigned int size;
  double learning_rate;
  double gamma[4];
  unsigned int num_inputs;
  unsigned int num_outputs;
};


const char *pycann_get_error(void);
void pycann_reset_error(void);

pycann_t *pycann_new(unsigned int size, unsigned int num_inputs, unsigned int num_outputs, unsigned int num_threads);
void pycann_del(pycann_t *net);

unsigned int pycann_is_threading_enabled(void);
unsigned int pycann_get_memory_usage(pycann_t *net);
unsigned int pycann_get_size(pycann_t *net);
unsigned int pycann_get_num_threads(pycann_t *net);

pycann_float_t pycann_get_learning_rate(pycann_t *net);
void pycann_set_learning_rate(pycann_t *net, pycann_float_t v);

pycann_float_t pycann_get_gamma(pycann_t *net, unsigned int i);
void pycann_set_gamma(pycann_t *net, unsigned int i, pycann_float_t v);

pycann_float_t pycann_get_weight(pycann_t *net, unsigned int i, unsigned int j);
void pycann_set_weight(pycann_t *net, unsigned int i, unsigned int j, pycann_float_t v);
void pycann_set_random_weights(pycann_t *net, pycann_float_t connection_rate);

pycann_float_t pycann_get_threshold(pycann_t *net, unsigned int i);
void pycann_set_threshold(pycann_t *net, unsigned int i, pycann_float_t v);

pycann_float_t pycann_get_activation(pycann_t *net, unsigned int i);
void pycann_set_activation(pycann_t *net, unsigned int i, pycann_float_t v);

unsigned int pycann_get_mod_neuron(pycann_t *net, unsigned int i);
pycann_float_t pycann_get_mod_weight(pycann_t *net, unsigned int i);
void pycann_set_mod(pycann_t *net, unsigned int i, unsigned int j, pycann_float_t weight);

unsigned int pycann_get_num_inputs(pycann_t *net);
unsigned int pycann_get_num_outputs(pycann_t *net);
void pycann_set_inputs(pycann_t *net, pycann_float_t *inputs);
void pycann_get_outputs(pycann_t *net, pycann_float_t *outputs);

void pycann_step(pycann_t *net, unsigned int n);

pycann_t *pycann_load_file(const char *path, unsigned int num_threads);
int pycann_save_file(const char *path, pycann_t *net);

#endif /* _PYCANN_H_ */
