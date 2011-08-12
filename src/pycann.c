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
#include <stdio.h> /* vsnprintf, fopen, fclose, fread, fwrite */
#include <string.h> /* memcpy */
#include <stdint.h> /* uint16_t, uint32_t */
#include <math.h> /* exp */

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
  net->gammas = pycann_malloc(net, sizeof(pycann_float_t)*4*size);
  net->weights = pycann_malloc(net, sizeof(pycann_float_t)*size*size);
  net->thresholds = pycann_malloc(net, sizeof(pycann_float_t)*size);
  net->activations = pycann_malloc(net, sizeof(pycann_float_t)*size);
  net->activation_functions = pycann_malloc(net, sizeof(pycann_activation_function_t)*size);
  net->mod_neurons = pycann_malloc(net, sizeof(pycann_float_t*)*size);
  net->mod_weights = pycann_malloc(net, sizeof(pycann_float_t)*size);
  net->inputs = pycann_malloc(net, sizeof(pycann_float_t)*num_inputs);

  // set values
  net->size = size;
  net->learning_rate = 0.0;
  net->num_inputs = num_inputs;
  net->num_outputs = num_outputs;

  // init weights, thresholds, activations and modularity connections
  for (i=0; i<size; i=i+1) {
    for (j=0; j<4; j++) {
      PYCANN_GAMMA(net, i, j) = 0.0;
    }
    for (j=0; j<size; j=j+1) {
      PYCANN_WEIGHT(net, i, j) = 0.0;
    }
    net->thresholds[i] = 0.0;
    net->activations[i] = 0.0;
    net->activation_functions[i] = PYCANN_SIGMOID_STEP;
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

  free(net->gammas);
  free(net->weights);
  free(net->thresholds);
  free(net->activations);
  free(net->mod_neurons);
  free(net->mod_weights);
  free(net->inputs);
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

// Get activation function
pycann_activation_function_t pycann_get_activation_function(pycann_t *net, unsigned int i) {
  if (i<net->size) {
    return net->activation_functions[i];
  }
  else {
    return 0;
  }
}
// Set activation function
void pycann_set_activation_function(pycann_t *net, unsigned int i, pycann_activation_function_t activation_function) {
  if (i<net->size) {
    net->activation_functions[i] = activation_function;
  }
}




// Get gamma
void pycann_get_gamma(pycann_t *net, unsigned int i, pycann_float_t *gamma) {
  if (i<net->size) {
    memcpy(gamma, net->gammas+(i*4), 4*sizeof(pycann_float_t));
  }
  else {
    pycann_set_error("Invalid neuron index: %d", i);
  }
}
// Set gamma
void pycann_set_gamma(pycann_t *net, unsigned int i, pycann_float_t *gamma) {
  if (i<net->size) {
    memcpy(net->gammas+(i*4), gamma, 4*sizeof(pycann_float_t));
  }
  else {
    pycann_set_error("Invalid neuron index: %d", i);
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
// FIXME: Definetely does NOT work!
void pycann_set_random_weights(pycann_t *net, pycann_float_t connection_rate) {
  unsigned int i, j, n;
  pycann_float_t sign;

  if (connection_rate>=0.0 && connection_rate<=1.0) {
    n = (int)(connection_rate*net->size);
    for (i=0; i<net->size; i=i+1) {
      for (j=0; j<n; j=j+1) {
	sign = rand()&1?+1.0:-1.0;
        PYCANN_WEIGHT(net, i, j) = sign * (pycann_float_t)(((double)rand())/((double)RAND_MAX));
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
    return net->mod_neurons[i]-net->activations;
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
  unsigned int i, o;

  o = net->size-net->num_outputs;
  for (i=0; i<net->num_outputs; i=i+1) {
    outputs[i] = net->activations[o+i];
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

// Sigmoidal approximation
static pycann_float_t pycann_sigmoid_approx(pycann_float_t x) {
  return x>=0.0?1.0:0.0; // TODO
}

// Internals of a neuron (propagation and activation function)
static pycann_float_t pycann_neuron_internal(pycann_t *net, unsigned int i) {
  unsigned int j;
  pycann_float_t o, u, v, w, dw, m, mw, t;

  t = net->thresholds[i];

  // propagation
  if (i<net->num_inputs) {
    o = net->inputs[i];
  }
  else {
    u = net->activations[i];
    o = 0.0;
    mw = net->mod_weights[i];
    m = (*net->mod_neurons[i]) * mw * net->learning_rate;

    for (j=0; j<net->size; j=j+1) {
      w = PYCANN_WEIGHT(net, i, j);
      v = net->activations[j];
      o = o+w*v;

      if (m!=0.0) {
        dw = (signbit(w)?-1.0:1.0) * m * (PYCANN_GAMMA(net, i, 0)*u*v + PYCANN_GAMMA(net, i, 1)*v + PYCANN_GAMMA(net, i, 2)*u + PYCANN_GAMMA(net, i, 3));
        PYCANN_WEIGHT(net, i, j) = w+dw;
      }
    }
  }

  // activations
  switch (net->activation_functions[i]) {
    case PYCANN_SIGMOID_STEP:
      return o>=t?1.0:0.0;
    case PYCANN_SIGMOID_EXP:
      return 1.0/(1.0+exp(PYCANN_SIGMOID_BETA*(t-o)));
    case PYCANN_SIGMOID_APPROX:
      return pycann_sigmoid_approx(t-o);
    case PYCANN_LINEAR:
      return o>1.0?1.0:(o<-0.0?0.0:o);
    default:
      return 0.0;
  }
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

// Loads network from pycann format file
// File extension .pcn
pycann_t *pycann_load_file(const char *path, unsigned int num_threads) {
  pycann_t *net;
  FILE *fd;
  unsigned int i;
  unsigned int *mod_neurons;
  struct pycann_file_header header;

  // open file
  fd = fopen(path, "rb");
  if (fd==NULL) {
    pycann_set_error("Can't open file (for reading): %s\n", path);
    return NULL;
  }

  // load header
  fread(&header, sizeof(header), 1, fd);
  if (memcmp(header.magic, PYCANN_FILE_MAGIC, PYCANN_FILE_MAGIC_LENGTH)!=0) {
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
  net->learning_rate = header.learning_rate;

  // load weights, etc.
  fread(net->gammas, 4*sizeof(pycann_float_t), header.size, fd);
  fread(net->weights, sizeof(pycann_float_t), header.size*header.size, fd);
  fread(net->thresholds, sizeof(pycann_float_t), header.size, fd);
  fread(net->activations, sizeof(pycann_float_t), header.size, fd);
  fread(net->mod_weights, sizeof(pycann_float_t), header.size, fd);
  fread(net->inputs, sizeof(pycann_float_t), header.num_inputs, fd);
  fread(net->activation_functions, sizeof(pycann_activation_function_t), net->size, fd);
  // load mod_neurons
  mod_neurons = malloc(sizeof(unsigned int)*header.size);
  fread(mod_neurons, sizeof(unsigned int), header.size, fd);
  for (i=0; i<header.size; i++) {
    net->mod_neurons[i] = net->activations+mod_neurons[i];
  }
  free(mod_neurons);

  // close file
  fclose(fd);

  return net;
}

// Saves network into the pycann format file
// File extension .pcn
int pycann_save_file(const char *path, pycann_t *net) {
  FILE *fd;
  struct pycann_file_header header;
  unsigned int i;
  unsigned int *mod_neurons;

  // open file
  fd = fopen(path, "w");
  if (fd==NULL) {
    pycann_set_error("Can't open file (for writing): %s\n", path);
    return -1;
  }

  // fill in header
  memcpy(header.magic, PYCANN_FILE_MAGIC, PYCANN_FILE_MAGIC_LENGTH);
  header.size = net->size;
  header.learning_rate = net->learning_rate;
  header.num_inputs = net->num_inputs;
  header.num_outputs = net->num_outputs;
  fwrite(&header, sizeof(header), 1, fd);

  // write weights, etc.
  fwrite(net->gammas, 4*sizeof(pycann_float_t), net->size, fd);
  fwrite(net->weights, sizeof(pycann_float_t), net->size*net->size, fd);
  fwrite(net->thresholds, sizeof(pycann_float_t), net->size, fd);
  fwrite(net->activations, sizeof(pycann_float_t), net->size, fd);
  fwrite(net->mod_weights, sizeof(pycann_float_t), net->size, fd);
  fwrite(net->inputs, sizeof(pycann_float_t), net->num_inputs, fd);
  fwrite(net->activation_functions, sizeof(pycann_activation_function_t), net->size, fd);
  // write mod neurons
  mod_neurons = malloc(sizeof(unsigned int)*net->size);
  for (i=0; i<header.size; i++) {
    mod_neurons[i] = net->mod_neurons[i]-net->activations;
  }
  fwrite(mod_neurons, sizeof(unsigned int), net->size, fd);
  free(mod_neurons);

  // close file
  fclose(fd);

  return 0;
}

// Exports into the pycann embedded format
// File extension: .rnn
// TODO export gamma and learning rate
int pycann_export_embedded(const char *path, pycann_t *net, int format) {
  FILE *fd;
  unsigned int i;
  uint16_t tmp;

  // TODO check if net is exportable

  // currently only NXT is supported
  if (format!=PYCANN_EMBEDDED_NXT) {
    return -1;
  }

  // open output file
  fd = fopen(path, "w");
  if (fd==NULL) {
    pycann_set_error("Can't open file (for writing): %s\n", path);
    return -1;
  }

  // write signature
  fwrite("RN", 1, 2, fd);
  // write format version
  fputc(PYCANN_EMBEDDED_VERSION, fd);
  // write network size
  tmp = net->size;
  fwrite(&tmp, sizeof(tmp), 1, fd);
  // write number of inputs
  tmp = net->num_inputs;
  fwrite(&tmp, sizeof(tmp), 1, fd);
  // write number of outputs
  tmp = net->num_outputs;
  fwrite(&tmp, sizeof(tmp), 1, fd);
  // write learning rate and gamma
  fwrite(&net->learning_rate, sizeof(pycann_float_t), 1, fd);

  // write gammas, thresholds and weights
  fwrite(net->gammas, 4*sizeof(pycann_float_t), net->size, fd);
  fwrite(net->weights, sizeof(pycann_float_t), net->size*net->size, fd);
  fwrite(net->thresholds, sizeof(pycann_float_t), net->size, fd);
  // write activation functions
  for (i=0; i<net->size; i=i+1) {
    fputc(net->activation_functions[i], fd);
  }

  // clean up
  fclose(fd);

  return 0;
}
