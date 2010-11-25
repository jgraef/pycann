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
#include <unistd.h> /* sleep */
#endif /* PYCANN_THREADING */

#include "pycann.h"


// Prototypes of static functions
// TODO add remaining
static void pycann_single_step(pycann_t *net, unsigned int first, unsigned int last);


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


// pycann's malloc functions (keeps track of used memory)
static void *pycann_malloc(pycann_t *net, unsigned int n) {
  net->memory_usage += n;
  return malloc(n);
}


#ifdef PYCANN_THREADING
static void pycann_thread_yield(void) {
  sleep(0);
}

// Thread main function
static void *pycann_thread_main(void *param) {
  pycann_t *net = (pycann_t*)param;
  pthread_t thread;
  pycann_thread_t *self = NULL;
  unsigned int n, i;

  unsigned int self_id;

  // Find own thread structure
  thread = pthread_self();
  for (i=0; i<net->num_threads; i=i+1) {
    if (pthread_equal(thread, net->threads[i].thread)) {
      self = net->threads+i;
      break;
    }
  }
  if (self==NULL) {
    pycann_set_error("THREAD INTERNAL ERROR: %s:%d\n", __FILE__, __LINE__);
    return NULL;
  }

  self_id = i;

  while (1) {
    // Cancellation point
    pthread_testcancel();

    n = self->steps;

    if (n==0) {
      // If nothing to do: sleep shortly to give CPU a rest
      // this must be longer, if thread really idles
      pycann_thread_yield();
    }
    else {
      // Else do steps
      for (i=0; i<n; i=i+1) {
        pycann_single_step(net, self->first_neuron, self->last_neuron);
      }

      // Done. Reset 'steps' to 0
      self->steps = 0;
    }
  }

  return NULL;
}
#endif /* PYCANN_THREADING */

// Create new network
pycann_t *pycann_new(unsigned int size, unsigned int num_inputs, unsigned int num_outputs, unsigned int num_threads) {
  pycann_t *net;
  unsigned int i, j, s, r;

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
    net->mod_neurons[i] = net->activations;
    net->mod_weights[i] = 0.0;
  }

  // init inputs
  for (i=0; i<num_inputs; i=i+1) {
    net->inputs[i] = 0.0;
  }

#ifdef PYCANN_THREADING
  // Initialize threading
  if (num_threads==0) {
    num_threads = 1;
  }
  net->num_threads = num_threads;
  net->threads = pycann_malloc(net, sizeof(pycann_thread_t)*num_threads);
  s = net->size/num_threads;
  r = net->size%num_threads;
  pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
  pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, NULL);

  // Main thread
  net->threads[0].thread = pthread_self();
  net->threads[0].first_neuron = 0;
  net->threads[0].last_neuron = s+r;
  net->threads[0].steps = 0;

  // Child threads
  for (i=1; i<num_threads; i=i+1) {
    net->threads[i].first_neuron = i*s+r;
    net->threads[i].last_neuron = (i+1)*s+r;
    net->threads[i].steps = 0;

    if (pthread_create(&(net->threads[i].thread), NULL, pycann_thread_main, net)!=0) {
      net->num_threads = 1;
      net->threads[0].last_neuron = net->size;
      pycann_set_error("Could not initialize thread #%d. Disabled multi-threading.\n", i);
      break;
    }
  }

  printf("Neuron disposal:\n");
  for (i=0; i<num_threads; i=i+1) {
    printf("  Thread #%d: [% 5d, % 5d)\n", i, net->threads[i].first_neuron, net->threads[i].last_neuron);
  }
#endif /* PYCANN_THREADING */

  return net;
}

// Delete network
void pycann_del(pycann_t *net) {
#ifdef PYCANN_THREADING
  // Terminate all threads
  unsigned int i;

  for (i=1; i<net->num_threads; i=i+1) {
    pthread_cancel(net->threads[i].thread);
    pthread_join(net->threads[i].thread, NULL);
  }

  free(net->threads);
#endif /* PYCANN_THREADING */

  free(net->weights);
  free(net->thresholds);
  free(net->activations);
  free(net);
}

// Get whether threading is enabled
unsigned int pycann_is_threading_enabled(void) {
#ifdef PYCANN_THREADING
  return 1;
#else
  return 0;
#endif /* PYCANN_THREADING */
}

// Get memory usage
unsigned int pycann_get_memory_usage(pycann_t *net) {
  return net->memory_usage;
}

// Get network size
unsigned int pycann_get_size(pycann_t *net) {
  return net->size;
}

// Get number of threads
unsigned int pycann_get_num_threads(pycann_t *net) {
#ifdef PYCANN_THREADING
  return net->num_threads;
#else
  return 1;
#endif /* PYCANN_THREADING */
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
// Set random weights
void pycann_set_random_weights(pycann_t *net, pycann_float_t connection_rate) {
  unsigned int i, j, n;

  if (connection_rate>=0.0 && connection_rate<=1.0) {
    n = (int)(connection_rate*net->size);
    for (i=0; i<net->size; i=i+1) {
      for (j=0; j<n; j=j+1) {
        PYCANN_WEIGHT(net, i, j) = (pycann_float_t)(((double)rand())/((double)RAND_MAX));
      }
    }
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

// Set inputs
void pycann_set_inputs(pycann_t *net, pycann_float_t *inputs) {
  memcpy(net->inputs, inputs, sizeof(pycann_float_t)*net->num_inputs);
}
// Get outputs
void pycann_get_outputs(pycann_t *net, pycann_float_t *outputs) {
  unsigned int i;

  for (i=net->size-net->num_outputs; i<net->size; i=i+1) {
    outputs[i] = net->activations[i];
  }
}
// Get number of inputs
unsigned int pycann_get_num_inputs(pycann_t *net) {
  return net->num_inputs;
}
// Get number of outputs
unsigned int pycann_get_num_outputs(pycann_t *net) {
  return net->num_outputs;
}

// Internals of a neuron (propagation and activation function)
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

  return o>t?1.0:0.001; // change 0.1 to 0.0 later
}

// Do a single step in a neural network (from neuron 'first' upto (excluding) neuron 'last')
static void pycann_single_step(pycann_t *net, unsigned int first, unsigned int last) {
  unsigned int i;

  for (i=first; i<last; i=i+1) {
    net->activations[i] = pycann_neuron_internal(net, i);
  }
}

// Do 'n' steps in a neural network (work is split between threads)
void pycann_step(pycann_t *net, unsigned int n) {
#ifdef PYCANN_THREADING
  unsigned int i, s;
  pycann_thread_t *self = net->threads;

  // tell  other threads number of steps (n)
  for (i=1; i<net->num_threads; i=i+1) {
    net->threads[i].steps = n;
  }

  // calculate main threads part
  for (s=0; s<n; s=s+1) {
    pycann_single_step(net, self->first_neuron, self->last_neuron);
  }

  // done
  self->steps = 0;

  // wait for other threads to finish, before returning
  for (i=1; i<net->num_threads; i=i+1) {
    while (net->threads[i].steps>0) {
      pycann_thread_yield();
    }
  }
#else
  unsigned int s;

  for (s=0; s<n; s=s+1) {
    pycann_single_step(net, 0, net->size);
  }
#endif /* PYCANN_THREADING */
}

// Load network from file
pycann_t *pycann_load_file(const char *path, unsigned int num_threads) {
  pycann_t *net;
  FILE *fd;
  unsigned int i, j;
  struct pycann_file_header header;
  pycann_float_t *thresholds;
  pycann_float_t *activations;
  unsigned int *mod_neurons;
  pycann_float_t *mod_weights;
  pycann_float_t *weights;
  pycann_float_t *inputs;

  // open file
  fd = fopen(path, "rb");
  if (fd==NULL) {
    pycann_set_error("Can't open file: %s\n", path);
    return NULL;
  }

  // load header
  fread(&header, sizeof(header), 1, fd);
  if (memcmp(header.signature, PYCANN_FILE_SIGNATURE, PYCANN_FILE_SIGNATURE_LENGTH)!=0) {
    pycann_set_error("Invalid file signature: %s\n", path);
    fclose(fd);
    return NULL;
  }

  // create ANN from header information
  net = pycann_new(header.size, header.num_inputs, header.num_outputs, num_threads);
  if (net==NULL) {
    fclose(fd);
    return NULL;
  }
  net->learning_rate = (pycann_float_t)header.learning_rate;
  for (i=0; i<4; i=i+1) {
    net->gamma[i] = (pycann_float_t)header.gamma[i];
  }

  // create buffers and load data
  thresholds = malloc(sizeof(pycann_float_t)*header.size);
  fread(thresholds, sizeof(pycann_float_t), header.size, fd);
  activations = malloc(sizeof(pycann_float_t)*header.size);
  fread(activations, sizeof(pycann_float_t), header.size, fd);
  mod_neurons = malloc(sizeof(unsigned int)*header.size);
  fread(mod_neurons, sizeof(unsigned int), header.size, fd);
  mod_weights = malloc(sizeof(pycann_float_t)*header.size);
  fread(mod_weights, sizeof(pycann_float_t), header.size, fd);
  weights = malloc(sizeof(pycann_float_t)*header.size*header.size);
  fread(weights, sizeof(pycann_float_t), header.size*header.size, fd);
  inputs = malloc(sizeof(pycann_float_t)*header.num_inputs);
  fread(inputs, sizeof(pycann_float_t), header.num_inputs, fd);

  // load thresholds, activations and weights
  for (i=0; i<header.size; i=i+1) {
    net->thresholds[i] = (pycann_float_t)thresholds[i];
    net->activations[i] = (pycann_float_t)activations[i];
    pycann_set_mod(net, i, mod_neurons[i], (pycann_float_t)mod_weights[i]);
    for (j=0; j<header.size; j=j+1) {
      PYCANN_WEIGHT(net, i, j) = (pycann_float_t)weights[i*header.size+j];
    }
  }

  // load input data
  for (i=0; i<header.num_inputs; i=i+1) {
    net->inputs[i] = (pycann_float_t)inputs[i];
  }

  // free buffers
  free(thresholds);
  free(activations);
  free(mod_neurons);
  free(mod_weights);
  free(weights);
  free(inputs);

  return net;
}

int pycann_save_file(const char *path, pycann_t *net) {
  FILE *fd;
  struct pycann_file_header header;
  unsigned int i, j;
  pycann_float_t *thresholds;
  pycann_float_t *activations;
  unsigned int *mod_neurons;
  pycann_float_t *mod_weights;
  pycann_float_t *weights;
  pycann_float_t *inputs;

  // open file
  fd = fopen(path, "w");
  if (fd==NULL) {
    pycann_set_error("Can't open file: %s\n", path);
    return -1;
  }

  // fill in header
  memcpy(header.signature, PYCANN_FILE_SIGNATURE, PYCANN_FILE_SIGNATURE_LENGTH);
  header.size = net->size;
  header.learning_rate = (double)net->learning_rate;
  for (i=0; i<4; i=i+1) {
    header.gamma[i] = (double)net->gamma[i];
  }
  header.num_inputs = net->num_inputs;
  header.num_outputs = net->num_outputs;
  fwrite(&header, sizeof(header), 1, fd);

  // create buffers
  thresholds = malloc(sizeof(pycann_float_t)*net->size);
  activations = malloc(sizeof(pycann_float_t)*net->size);
  mod_neurons = malloc(sizeof(unsigned int)*net->size);
  mod_weights = malloc(sizeof(pycann_float_t)*net->size);
  weights = malloc(sizeof(pycann_float_t)*net->size*net->size);
  inputs = malloc(sizeof(pycann_float_t)*net->num_inputs);

  // fill in buffers
  for (i=0; i<net->size; i=i+1) {
    thresholds[i] = (double)net->thresholds[i];
    activations[i] = (double)net->activations[i];
    pycann_set_mod(net, i, mod_neurons[i], (pycann_float_t)mod_weights[i]);
    mod_neurons[i] = (double)pycann_get_mod_neuron(net, i);
    mod_weights[i] = (double)net->mod_weights[i];
    for (j=0; j<net->size; j=j+1) {
      weights[i*net->size+j] = (double)PYCANN_WEIGHT(net, i, j);
    }
  }
  for (i=0; i<net->num_inputs; i=i+1) {
    inputs[i] = (double)net->inputs[i];
  }

  // write & free buffers
  fwrite(thresholds, sizeof(pycann_float_t), net->size, fd);
  free(thresholds);
  fwrite(activations, sizeof(pycann_float_t), net->size, fd);
  free(activations);
  fwrite(mod_neurons, sizeof(unsigned int), net->size, fd);
  free(mod_neurons);
  fwrite(mod_weights, sizeof(pycann_float_t), net->size, fd);
  free(mod_weights);
  fwrite(weights, sizeof(pycann_float_t), net->size*net->size, fd);
  free(weights);
  fwrite(inputs, sizeof(pycann_float_t), net->num_inputs, fd);
  free(weights);

  return 0;
}