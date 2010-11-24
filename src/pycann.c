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

#include <stdlib.h> /* malloc, free */
#include <stdarg.h> /* va_list, va_start, va_end */
#include <stdio.h> /* vsnprintf */
#include <string.h> /* memcpy */

#ifdef PYCANN_THREADING
#include <pthread.h>
#endif /* PYCANN_THREADING */

#include "pycann.h"

// Buffer for current error
#define PYCANN_ERROR_SIZE 1024
static char pycann_error[PYCANN_ERROR_SIZE];


// Get current error
const char *pycann_get_error(void) {
  return pycann_error;
}
// Reset current error
void pycann_reset_error(void) {
  pycann_error[0] = 0;
}
// Set current error
static void pycann_set_error(const char *fmt, ...) {
  va_list ap;

  va_start(ap, fmt);
  vsnprintf(pycann_error, PYCANN_ERROR_SIZE, fmt, ap);
  va_end(ap);
}

#define DEBUG printf("debug: %s:%d (%s)\n", __FILE__, __LINE__, __func__);

// pycann's malloc functions (keeps track of used memory)
void *pycann_malloc(pycann_t *net, unsigned int n) {
  net->memory_usage += n;
  return malloc(n);
}

// Create new network
pycann_t *pycann_new(unsigned int size, unsigned int num_inputs, unsigned int num_outputs, unsigned int num_threads) {
  pycann_t *net;
  unsigned int i, j;

  // allocate memory
  net = malloc(sizeof(pycann_t));
  net->memory_usage = sizeof(pycann_t);
  net->weights = pycann_malloc(net, sizeof(pycann_float_t)*size*size);
  net->thresholds = pycann_malloc(net, sizeof(pycann_float_t)*size);
  net->activations = pycann_malloc(net, sizeof(pycann_float_t)*size);
  net->mod_neurons = pycann_malloc(net, sizeof(unsigned int)*size);
  net->mod_weights = pycann_malloc(net, sizeof(pycann_float_t)*size);
  net->inputs = pycann_malloc(net, sizeof(pycann_float_t)*num_outputs);

  // set values
  net->size = size;
  net->learning_rate = 0.0;
  net->gamma[0] = 0.0;
  net->gamma[1] = 0.0;
  net->gamma[2] = 0.0;
  net->gamma[3] = 0.0;
  net->num_inputs = num_inputs;
  net->num_outputs = num_outputs;

  // init weights, thresholds, activations and modularity connections
  for (i=0; i<size; i=i+1) {
    for (j=0; j<size; j=j+1) {
      PYCANN_WEIGHT(net, i, j) = 0.0;
    }
    net->thresholds[i] = 0.0;
    net->activations[i] = 0.0;
    net->mod_neurons[i] = 0;
    net->mod_weights[i] = 0.0;
  }

  // init inputs
  for (i=0; i<num_inputs; i=i+1) {
    net->inputs[i] = 0.0;
  }

#ifdef PYCANN_THREADING
  // Threading
  if (num_threads==0) {
    num_threads = 1;
  }
  net->num_threads = num_threads;
  net->threads = pycann_malloc(net, sizeof(pthread_t)*num_threads);

  for (i=0; i<num_threads; i=i+1) {
    if (pthread_create(net->threads+i, NULL, pycann_thread_main, net)!=0) {
      net->num_threads = 1;
      pycann_set_error("Could not initialize thread #%d. Disabled multi-threading.\n", i);
    }
  }
#endif /* PYCANN_THREADING */

  return net;
}

// Delete network
void pycann_del(pycann_t *net) {
  free(net->weights);
  free(net->thresholds);
  free(net->activations);
  free(net);
}

// Get memory usage
unsigned int pycann_get_memory_usage(pycann_t *net) {
  return net->memory_usage;
}

// Get network size
unsigned int pycann_get_size(pycann_t *net) {
  return net->size;
}

// Get learning rate
pycann_float_t pycann_get_learning_rate(pycann_t *net) {
  return net->learning_rate;
}
// Set learning rate
void pycann_set_learning_rate(pycann_t *net, pycann_float_t v) {
  net->learning_rate = v;
}

// Get gamma i
pycann_float_t pycann_get_gamma(pycann_t *net, unsigned int i) {
  if (i<4) {
    return net->gamma[i];
  }
  else {
    pycann_set_error("Invalid gamma index: %d", i);
    return 0.0;
  }
}
// Set gamma i
void pycann_set_gamma(pycann_t *net, unsigned int i, pycann_float_t v) {
  if (i<4) {
    net->gamma[i] = v;
  }
}

// Get weight
pycann_float_t pycann_get_weight(pycann_t *net, unsigned int i, unsigned int j) {
  if (i<net->size && j<net->size) {
    return PYCANN_WEIGHT(net, i, j);
  }
  else {
    return 0.0;
  }
}
// Set weight
void pycann_set_weight(pycann_t *net, unsigned int i, unsigned int j, pycann_float_t v) {
  if (i<net->size && j<net->size) {
    PYCANN_WEIGHT(net, i, j) = v;
  }
}

// Get threshold
pycann_float_t pycann_get_threshold(pycann_t *net, unsigned int i) {
  if (i<net->size) {
    return net->thresholds[i];
  }
  else {
    return 0.0;
  }
}
// Set threshold
void pycann_set_threshold(pycann_t *net, unsigned int i, pycann_float_t v) {
  if (i<net->size) {
    net->thresholds[i] = v;
  }
}

// Get activation
pycann_float_t pycann_get_activation(pycann_t *net, unsigned int i) {
  if (i<net->size) {
    return net->activations[i];
  }
  else {
    return 0.0;
  }
}
// Set activation
void pycann_set_activation(pycann_t *net, unsigned int i, pycann_float_t v) {
  if (i<net->size) {
    net->activations[i] = v;
  }
}

// Get modularity neuron
unsigned int pycann_get_mod_neuron(pycann_t *net, unsigned int i) {
  if (i<net->size) {
    // TODO convert pointer to index
    return 0;
    //return net->mod_neurons[i];
  }
  else {
    return 0;
  }
}
// Get modularity weight
pycann_float_t pycann_get_mod_weight(pycann_t *net, unsigned int i) {
  if (i<net->size) {
    return net->mod_weights[i];
  }
  else {
    return 0.0;
  }
}
// Set modularity connection
void pycann_set_mod(pycann_t *net, unsigned int i, unsigned int j, pycann_float_t weight) {
  if (i<net->size && j<net->size) {
    net->mod_neurons[i] = net->activations+j;
    net->mod_weights[i] = net->learning_rate*weight;
  }
}

void pycann_set_inputs(pycann_t *net, pycann_float_t *inputs) {
  memcpy(net->inputs, inputs, sizeof(pycann_float_t)*net->num_inputs);
}

void pycann_get_outputs(pycann_t *net, pycann_float_t *outputs) {
  unsigned int i;

  for (i=net->size-net->num_outputs; i<net->size; i=i+1) {
    outputs[i] = net->activations[i];
  }
}

static pycann_float_t pycann_neuron_internal(pycann_t *net, unsigned int i) {
  unsigned int j;
  pycann_float_t o, u, v, w, dw, m, mw, t;

  t = net->thresholds[i];

  if (i<net->num_inputs) {
    o = net->inputs[i];
  }
  else {
    u = net->activations[i];
    o = 0.0;
    mw = net->mod_weights[i];
    m = (*net->mod_neurons[i])*mw;

    for (j=0; j<net->size; j=j+1) {
      w = PYCANN_WEIGHT(net, i, j);
      v = net->activations[j];
      o = o+w*v;

      if (m!=0.0) {
        dw = m*(net->gamma[0]*u*v + net->gamma[1]*u + net->gamma[2]*v + net->gamma[4]);
        PYCANN_WEIGHT(net, i, j) = w+dw;
      }
      else if (o>t) {
        // we can abort since threshold is already exceeded and no learning is done
        return 1.0;
      }
    }
  }

  return o>t?1.0:0.1; // change 0.1 to 0.0 later
}

void pycann_step(pycann_t *net, unsigned int n) {
  unsigned int s;
  unsigned int i;
  pycann_float_t a;

  for (s=0; s<n; s=s+1) {
    for (i=0; i<net->size; i=i+1) {
      net->activations[i] = pycann_neuron_internal(net, i);
    }
  }
}
