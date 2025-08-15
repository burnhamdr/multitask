"""Definition of the network model and various RNN cells"""

from __future__ import division

import os
import numpy as np
import pickle

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import nest
from tensorflow.python.ops import rnn
from tensorflow.python.ops.rnn_cell_impl import RNNCell

import tools


def is_weight(v):
    """Check if Tensorflow variable v is a connection weight."""
    return ('kernel' in v.name or 'weight' in v.name)

def popvec(y):
    """Population vector read out.

    Assuming the last dimension is the dimension to be collapsed

    Args:
        y: population output on a ring network. Numpy array (Batch, Units)

    Returns:
        Readout locations: Numpy array (Batch,)
    """
    pref = np.arange(0, 2*np.pi, 2*np.pi/y.shape[-1])  # preferences
    temp_sum = y.sum(axis=-1)
    temp_cos = np.sum(y*np.cos(pref), axis=-1)/temp_sum
    temp_sin = np.sum(y*np.sin(pref), axis=-1)/temp_sum
    loc = np.arctan2(temp_sin, temp_cos)
    #print(loc)
    return np.mod(loc, 2*np.pi)


def tf_popvec(y):
    """Population vector read-out in tensorflow."""

    num_units = y.get_shape().as_list()[-1]
    pref = np.arange(0, 2 * np.pi, 2 * np.pi / num_units)  # preferences
    cos_pref = np.cos(pref)
    sin_pref = np.sin(pref)
    temp_sum = tf.reduce_sum(y, axis=-1)
    temp_cos = tf.reduce_sum(y * cos_pref, axis=-1) / temp_sum
    temp_sin = tf.reduce_sum(y * sin_pref, axis=-1) / temp_sum
    loc = tf.atan2(temp_sin, temp_cos)
    return tf.mod(loc, 2*np.pi)


def get_perf(y_hat, y_loc):
    """Get performance.

    Args:
      y_hat: Actual output. Numpy array (Time, Batch, Unit)
      y_loc: Target output location (-1 for fixation).
        Numpy array (Time, Batch)

    Returns:
      perf: Numpy array (Batch,)
    """
    if len(y_hat.shape) != 3:
        raise ValueError('y_hat must have shape (Time, Batch, Unit)')
    # Only look at last time points
    y_loc = y_loc[-1]
    y_hat = y_hat[-1]

    # Fixation and location of y_hat
    y_hat_fix = y_hat[..., 0]
    y_hat_loc = popvec(y_hat[..., 1:])

    # Fixating? Correctly saccading?
    fixating = y_hat_fix > 0.5

    original_dist = y_loc - y_hat_loc
    dist = np.minimum(abs(original_dist), 2*np.pi-abs(original_dist))
    corr_loc = dist < 0.2*np.pi

    # Should fixate?
    should_fix = y_loc < 0

    # performance
    perf = should_fix * fixating + (1-should_fix) * corr_loc * (1-fixating)
    return perf


class LeakyRNNCell(RNNCell):
    """The most basic RNN cell.

    Args:
        num_units: int, The number of units in the RNN cell.
        activation: Nonlinearity to use.    Default: `tanh`.
        reuse: (optional) Python boolean describing whether to reuse variables
         in an existing scope.    If not `True`, and the existing scope already has
         the given variables, an error is raised.
        name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
    """

    def __init__(self,
                 num_units,
                 n_input,
                 alpha,
                 num_attractors=10,
                 sigma_rec=0,
                 activation='softplus',
                 w_rec_init='diag',
                 w_in_init='randgauss',
                 b_rec_init='zeros',
                 rng=None,
                 reuse=None,
                 name=None):
        super(LeakyRNNCell, self).__init__(_reuse=reuse, name=name)

        # Inputs must be 2-dimensional.
        # self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._num_inputs = n_input
        self._w_rec_init = w_rec_init
        self._w_in_init = w_in_init
        self._b_rec_init = b_rec_init
        self._reuse = reuse

        if activation == 'softplus':
            self._activation = tf.nn.softplus
            self._w_in_start = 1.0
            self._w_rec_start = 0.5#0.5
        elif activation == 'tanh':
            self._activation = tf.tanh
            self._w_in_start = 1.0
            self._w_rec_start = 1.0
        elif activation == 'relu':
            self._activation = tf.nn.relu
            self._w_in_start = 1.0
            self._w_rec_start = 0.5#0.5
        elif activation == 'power':
            self._activation = lambda x: tf.square(tf.nn.relu(x))
            self._w_in_start = 1.0
            self._w_rec_start = 0.01
        elif activation == 'retanh':
            self._activation = lambda x: tf.tanh(tf.nn.relu(x))
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        elif activation == 'sigmoid':
            self._activation = tf.nn.sigmoid
            self._w_in_start = 1.0
            self._w_rec_start = 1.0
        else:
            raise ValueError('Unknown activation')
        self._alpha = alpha
        self._sigma = np.sqrt(2 / alpha) * sigma_rec
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        # Generate initialization matrix
        n_hidden = self._num_units
        n_input = self._num_inputs

        if isinstance(self._w_in_init, np.ndarray):
            assert self._w_in_init.shape == (n_input, n_hidden)
            w_in0 = self._w_in_init
        else:
            if self._w_in_init == 'randgauss':
                w_in0 = (self.rng.randn(n_input, n_hidden) /
                        np.sqrt(n_input) * self._w_in_start)
            elif self._w_in_init == 'ones':
                w_in0 = (np.ones((n_input, n_hidden)) * self._w_in_start)

        if isinstance(self._w_rec_init, np.ndarray):
            assert self._w_rec_init.shape == (n_hidden, n_hidden)
            w_rec0 = self._w_rec_init
        else:
            if self._w_rec_init == 'diag':
                w_rec0 = self._w_rec_start*np.eye(n_hidden)
            elif self._w_rec_init == 'randortho':
                w_rec0 = self._w_rec_start*tools.gen_ortho_matrix(n_hidden,
                                                                rng=self.rng)
            elif self._w_rec_init == 'randgauss':
                w_rec0 = (self._w_rec_start *
                        self.rng.randn(n_hidden, n_hidden)/np.sqrt(n_hidden))
            elif self._w_rec_init == 'singlering':
                w_rec0 = create_recurrent_weights_ring_attractor(n_hidden, 0.5, 1)
            elif self._w_rec_init == 'lowranknoise':
                print(num_attractors)
                w_rec0 = init_bernoulli_lowrank_plus_noise(n_hidden, num_attractors, 0.5)
            elif self._w_rec_init == 'newsinglering':
                w_rec0 = new_create_recurrent_weights_ring_attractor(n_hidden, 2.1, 1.6, 2.)
            elif self._w_rec_init == 'newdoublering':
                w_rec0 = new_create_recurrent_weights_two_ring_attractor(n_hidden, 2.1, 1.6, 2.)
            elif self._w_rec_init == 'plane':
                w_rec0 = new_create_recurrent_weights_plane_attractor(n_hidden,2.1,1.6,2.0)

        matrix0 = np.concatenate((w_in0, w_rec0), axis=0)

        self.w_rnn0 = matrix0
        self._initializer = tf.constant_initializer(matrix0, dtype=tf.float32)

        if isinstance(self._b_rec_init, np.ndarray):
            assert self._b_rec_init.shape == (n_hidden,)
            self._bias_initializer = tf.constant_initializer(self._b_rec_init, dtype=tf.float32)
        else:
            if self._b_rec_init == 'zeros':
                self._bias_initializer = init_ops.zeros_initializer(dtype=tf.float32)
            elif self._b_rec_init == 'uniform':
                self._bias_initializer = init_ops.random_uniform_initializer(
                    minval=-0.1, maxval=0.1, dtype=tf.float32)
            elif self._b_rec_init == 'randgauss':
                self._bias_initializer = init_ops.random_normal_initializer(dtype=tf.float32)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError(
                "Expected inputs.shape[-1] to be known, saw shape: %s"
                                             % inputs_shape)

        input_depth = inputs_shape[1].value
        self._kernel = self.add_variable(
                'kernel',
                shape=[input_depth + self._num_units, self._num_units],
                initializer=self._initializer)
        self._bias = self.add_variable(
                'bias',
                shape=[self._num_units],
                initializer=self._bias_initializer)

        self.built = True

    def call(self, inputs, state):
        """Most basic RNN: output = new_state = act(W * input + U * state + B)."""

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        noise = tf.random_normal(tf.shape(state), mean=0, stddev=self._sigma)
        gate_inputs = gate_inputs + noise

        output = self._activation(gate_inputs)

        output = (1-self._alpha) * state + self._alpha * output

        return output, output


class LeakyGRUCell(RNNCell):
  """Leaky Gated Recurrent Unit cell (cf. https://elifesciences.org/articles/21492).

  Args:
    num_units: int, The number of units in the GRU cell.
    alpha: dt/T, simulation time step over time constant
    sigma_rec: recurrent noise
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
  """

  def __init__(self,
               num_units,
               alpha,
               sigma_rec=0,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               name=None):
    super(LeakyGRUCell, self).__init__(_reuse=reuse, name=name)

    # Inputs must be 2-dimensional.
    # self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

    self._alpha = alpha
    self._sigma = np.sqrt(2 / alpha) * sigma_rec

    # TODO(gryang): allow this to use different initialization

  @property
  def state_size(self):
      return self._num_units

  @property
  def output_size(self):
      return self._num_units

  def build(self, inputs_shape):
      if inputs_shape[1].value is None:
        raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                         % inputs_shape)

      input_depth = inputs_shape[1].value
      self._gate_kernel = self.add_variable(
          "gates/%s" % 'kernel',
          shape=[input_depth + self._num_units, 2 * self._num_units],
          initializer=self._kernel_initializer)
      self._gate_bias = self.add_variable(
          "gates/%s" % 'bias',
          shape=[2 * self._num_units],
          initializer=(
              self._bias_initializer
              if self._bias_initializer is not None
              else init_ops.constant_initializer(1.0, dtype=self.dtype)))
      self._candidate_kernel = self.add_variable(
          "candidate/%s" % 'kernel',
          shape=[input_depth + self._num_units, self._num_units],
          initializer=self._kernel_initializer)
      self._candidate_bias = self.add_variable(
          "candidate/%s" % 'bias',
          shape=[self._num_units],
          initializer=(
              self._bias_initializer
              if self._bias_initializer is not None
              else init_ops.zeros_initializer(dtype=self.dtype)))

      self.built = True

  def call(self, inputs, state):
      """Gated recurrent unit (GRU) with nunits cells."""

      gate_inputs = math_ops.matmul(
          array_ops.concat([inputs, state], 1), self._gate_kernel)
      gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

      value = math_ops.sigmoid(gate_inputs)
      r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

      r_state = r * state

      candidate = math_ops.matmul(
          array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
      candidate = nn_ops.bias_add(candidate, self._candidate_bias)
      candidate += tf.random_normal(tf.shape(state), mean=0, stddev=self._sigma)

      c = self._activation(candidate)
      # new_h = u * state + (1 - u) * c  # original GRU
      new_h = (1 - self._alpha * u) * state + (self._alpha * u) * c

      return new_h, new_h


class LeakyRNNCellSeparateInput(RNNCell):
    """The most basic RNN cell with external inputs separated.

    Args:
        num_units: int, The number of units in the RNN cell.
        activation: Nonlinearity to use.    Default: `tanh`.
        reuse: (optional) Python boolean describing whether to reuse variables
         in an existing scope.    If not `True`, and the existing scope already has
         the given variables, an error is raised.
        name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
    """

    def __init__(self,
                 num_units,
                 alpha,
                 sigma_rec=0,
                 activation='softplus',
                 w_rec_init='diag',
                 rng=None,
                 reuse=None,
                 name=None):
        super(LeakyRNNCellSeparateInput, self).__init__(_reuse=reuse, name=name)

        # Inputs must be 2-dimensional.
        # self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._w_rec_init = w_rec_init
        self._reuse = reuse

        if activation == 'softplus':
            self._activation = tf.nn.softplus
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        elif activation == 'relu':
            self._activation = tf.nn.relu
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        else:
            raise ValueError('Unknown activation')
        self._alpha = alpha
        self._sigma = np.sqrt(2 / alpha) * sigma_rec
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        # Generate initialization matrix
        n_hidden = self._num_units

        if self._w_rec_init == 'diag':
            w_rec0 = self._w_rec_start*np.eye(n_hidden)
        elif self._w_rec_init == 'randortho':
            w_rec0 = self._w_rec_start*tools.gen_ortho_matrix(n_hidden,
                                                              rng=self.rng)
        elif self._w_rec_init == 'randgauss':
            w_rec0 = (self._w_rec_start *
                      self.rng.randn(n_hidden, n_hidden)/np.sqrt(n_hidden))
        else:
            raise ValueError

        self.w_rnn0 = w_rec0
        self._initializer = tf.constant_initializer(w_rec0, dtype=tf.float32)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        self._kernel = self.add_variable(
                'kernel',
                shape=[self._num_units, self._num_units],
                initializer=self._initializer)
        self._bias = self.add_variable(
                'bias',
                shape=[self._num_units],
                initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, inputs, state):
        """output = new_state = act(input + U * state + B)."""

        gate_inputs = math_ops.matmul(state, self._kernel)
        gate_inputs = gate_inputs + inputs  # directly add inputs
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        noise = tf.random_normal(tf.shape(state), mean=0, stddev=self._sigma)
        gate_inputs = gate_inputs + noise

        output = self._activation(gate_inputs)

        output = (1-self._alpha) * state + self._alpha * output

        return output, output



class Model(object):
    """The model."""

    def __init__(self,
                 model_dir,
                 hp=None,
                 sigma_rec=None,
                 dt=None,
                 display_info=True):
        """
        Initializing the model with information from hp

        Args:
            model_dir: string, directory of the model
            hp: a dictionary or None
            sigma_rec: if not None, overwrite the sigma_rec passed by hp
        """

        # Reset tensorflow graphs
        tf.reset_default_graph()  # must be in the beginning

        if hp is None:
            hp = tools.load_hp(model_dir)
            if hp is None:
                raise ValueError(
                    'No hp found for model_dir {:s}'.format(model_dir))

        tf.set_random_seed(hp['seed'])
        self.rng = np.random.RandomState(hp['seed'])

        if sigma_rec is not None:
            if display_info:
                print('Overwrite sigma_rec with {:0.3f}'.format(sigma_rec))
            hp['sigma_rec'] = sigma_rec

        if dt is not None:
            if display_info:
                print('Overwrite original dt with {:0.1f}'.format(dt))
            hp['dt'] = dt

        if 'alpha' not in hp:
            hp['alpha'] = 1.0*hp['dt']/hp['tau']

        # Input, target output, and cost mask
        # Shape: [Time, Batch, Num_units]
        if hp['in_type'] != 'normal':
            raise ValueError('Only support in_type ' + hp['in_type'])

        self._build(hp)

        self.model_dir = model_dir
        self.hp = hp

    def _build(self, hp):
        if 'use_separate_input' in hp and hp['use_separate_input']:
            self._build_seperate(hp)
        else:
            self._build_fused(hp)

        self.var_list = tf.trainable_variables()
        self.weight_list = [v for v in self.var_list if is_weight(v)]

        if 'use_separate_input' in hp and hp['use_separate_input']:
            self._set_weights_separate(hp)
        else:
            self._set_weights_fused(hp)

        # Regularization terms
        self.cost_reg = tf.constant(0.)
        if hp['l1_h'] > 0:
            self.cost_reg += tf.reduce_mean(tf.abs(self.h)) * hp['l1_h']
        if hp['l2_h'] > 0:
            self.cost_reg += tf.nn.l2_loss(self.h) * hp['l2_h']

        if hp['l1_weight'] > 0:
            self.cost_reg += hp['l1_weight'] * tf.add_n(
                [tf.reduce_mean(tf.abs(v)) for v in self.weight_list])
        if hp['l2_weight'] > 0:
            self.cost_reg += hp['l2_weight'] * tf.add_n(
                [tf.nn.l2_loss(v) for v in self.weight_list])

        # Create an optimizer.
        if 'optimizer' not in hp or hp['optimizer'] == 'adam':
            self.opt = tf.train.AdamOptimizer(
                learning_rate=hp['learning_rate'])
        elif hp['optimizer'] == 'sgd':
            self.opt = tf.train.GradientDescentOptimizer(
                learning_rate=hp['learning_rate'])
        
        # Set cost
        self.set_optimizer()

        # Variable saver
        # self.saver = tf.train.Saver(self.var_list)
        self.saver = tf.train.Saver()

    def _build_fused(self, hp):
        n_input = hp['n_input']
        n_rnn = hp['n_rnn']
        n_output = hp['n_output']

        self.x = tf.placeholder("float", [None, None, n_input])
        self.y = tf.placeholder("float", [None, None, n_output])
        if hp['loss_type'] == 'lsq':
            self.c_mask = tf.placeholder("float", [None, n_output])
        else:
            # Mask on time
            self.c_mask = tf.placeholder("float", [None])

        # Activation functions
        if hp['activation'] == 'power':
            f_act = lambda x: tf.square(tf.nn.relu(x))
        elif hp['activation'] == 'retanh':
            f_act = lambda x: tf.tanh(tf.nn.relu(x))
        elif hp['activation'] == 'relu+':
            f_act = lambda x: tf.nn.relu(x + tf.constant(1.))
        else:
            f_act = getattr(tf.nn, hp['activation'])

        # Recurrent activity
        if hp['rnn_type'] == 'LeakyRNN':
            n_in_rnn = self.x.get_shape().as_list()[-1]

            # Check if all the necessary parameters are present, set default if not

            if 'w_in_init' not in hp:
                hp['w_in_init']='randgauss'
            if 'b_rec_init' not in hp:
                hp['b_rec_init']='zeros'
            if 'w_out_init' not in hp:
                hp['w_out_init']='glorot_uniform'
            if 'b_out_init' not in hp:
                hp['b_out_init']='zeros'
            if 'num_attractors' not in hp:
                hp['num_attractors']=10

            cell = LeakyRNNCell(n_rnn, n_in_rnn,
                                hp['alpha'],
                                hp['num_attractors'],
                                sigma_rec=hp['sigma_rec'],
                                activation=hp['activation'],
                                w_rec_init=hp['w_rec_init'],
                                w_in_init=hp['w_in_init'],
                                b_rec_init=hp['b_rec_init'],
                                rng=self.rng)
        elif hp['rnn_type'] == 'LeakyGRU':
            cell = LeakyGRUCell(
                n_rnn, hp['alpha'],
                sigma_rec=hp['sigma_rec'], activation=f_act)
        elif hp['rnn_type'] == 'LSTM':
            cell = tf.contrib.rnn.LSTMCell(n_rnn, activation=f_act)

        elif hp['rnn_type'] == 'GRU':
            cell = tf.contrib.rnn.GRUCell(n_rnn, activation=f_act)
        else:
            raise NotImplementedError("""rnn_type must be one of LeakyRNN,
                    LeakyGRU, EILeakyGRU, LSTM, GRU
                    """)

        # Dynamic rnn with time major
        self.h, states = rnn.dynamic_rnn(
            cell, self.x, dtype=tf.float32, time_major=True)

        # Output
        if isinstance(hp['w_out_init'], np.ndarray):
            assert hp['w_out_init'].shape == (n_rnn, n_output)
            self._w_out_initializer = tf.constant_initializer(hp['w_out_init'], dtype=tf.float32)
        elif hp['w_out_init'] == 'glorot_uniform':
                self._w_out_initializer = init_ops.glorot_uniform_initializer(dtype=tf.float32)
        elif hp['w_out_init'] == 'randgauss':
                self._w_out_initializer = init_ops.random_normal_initializer(dtype=tf.float32)

        if isinstance(hp['b_out_init'], np.ndarray):
            assert hp['b_out_init'].shape == (n_output,)
            self._b_out_initializer = tf.constant_initializer(hp['b_out_init'], dtype=tf.float32)
        else:
            if hp['b_out_init'] == 'zeros':
                self._b_out_initializer = init_ops.zeros_initializer(dtype=tf.float32)
            elif hp['b_out_init'] == 'uniform':
                self._b_out_initializer = init_ops.random_uniform_initializer(
                    minval=-0.1, maxval=0.1, dtype=tf.float32)
            elif hp['b_out_init'] == 'randgauss':
                self._b_out_initializer = init_ops.random_normal_initializer(dtype=tf.float32)

        with tf.variable_scope("output"):
            # Using default initialization `glorot_uniform_initializer`
            # w_out = tf.get_variable(
            #     'weights',
            #     [n_rnn, n_output],
            #     dtype=tf.float32,
            # )
            w_out = tf.get_variable(
                'weights',
                [n_rnn, n_output],
                dtype=tf.float32,
                initializer=self._w_out_initializer
            )
            b_out = tf.get_variable(
                'biases',
                [n_output],
                dtype=tf.float32,
                initializer=self._b_out_initializer
            )

        #self.h = tf.Print(self.h, [tf.shape(self.h)], message="Shape of h: ")

        h_shaped = tf.reshape(self.h, (-1, n_rnn))
        y_shaped = tf.reshape(self.y, (-1, n_output))

        #h_shaped = tf.Print(h_shaped, [tf.shape(h_shaped)], message="Shape of h_shaped: ")
        #y_shaped= tf.Print(y_shaped, [tf.shape(y_shaped)], message="Shape of y_shaped: ")
        #w_out= tf.Print(w_out, [tf.shape(w_out)], message="Shape of w_out: ")
        # y_hat_ shape (n_time*n_batch, n_unit)

        y_hat_ = tf.matmul(h_shaped, w_out) + b_out

        #y_hat_= tf.Print(y_hat_, [tf.shape(y_hat_)], message="Shape of y_hat_: ")

        if hp['loss_type'] == 'lsq':
            # Least-square loss
            y_hat = tf.sigmoid(y_hat_)
            self.cost_lsq = tf.reduce_mean(
                tf.square((y_shaped - y_hat) * self.c_mask))
        else:
            y_hat = tf.nn.softmax(y_hat_)
            # Cross-entropy loss
            self.cost_lsq = tf.reduce_mean(
                self.c_mask * tf.nn.softmax_cross_entropy_with_logits(
                    labels=y_shaped, logits=y_hat_))

        self.y_hat = tf.reshape(y_hat,
                                (-1, tf.shape(self.h)[1], n_output))
        y_hat_fix, y_hat_ring = tf.split(
            self.y_hat, [1, n_output - 1], axis=-1)
        self.y_hat_loc = tf_popvec(y_hat_ring)

    def _set_weights_fused(self, hp):
        """Set model attributes for several weight variables."""
        n_input = hp['n_input']
        n_rnn = hp['n_rnn']
        n_output = hp['n_output']

        for v in self.var_list:
            if 'rnn' in v.name:
                if 'kernel' in v.name or 'weight' in v.name:
                    # TODO(gryang): For GRU, fix
                    self.w_rec = v[n_input:, :]
                    self.w_in = v[:n_input, :]
                else:
                    self.b_rec = v
            else:
                assert 'output' in v.name
                if 'kernel' in v.name or 'weight' in v.name:
                    self.w_out = v
                else:
                    self.b_out = v

        # check if the recurrent and output connection has the correct shape
        if self.w_out.shape != (n_rnn, n_output):
            raise ValueError('Shape of w_out should be ' +
                             str((n_rnn, n_output)) + ', but found ' +
                             str(self.w_out.shape))
        if self.w_rec.shape != (n_rnn, n_rnn):
            raise ValueError('Shape of w_rec should be ' +
                             str((n_rnn, n_rnn)) + ', but found ' +
                             str(self.w_rec.shape))
        if self.w_in.shape != (n_input, n_rnn):
            raise ValueError('Shape of w_in should be ' +
                             str((n_input, n_rnn)) + ', but found ' +
                             str(self.w_in.shape))

    def _build_seperate(self, hp):
        # Input, target output, and cost mask
        # Shape: [Time, Batch, Num_units]
        n_input = hp['n_input']
        n_rnn = hp['n_rnn']
        n_output = hp['n_output']

        self.x = tf.placeholder("float", [None, None, n_input])
        self.y = tf.placeholder("float", [None, None, n_output])
        self.c_mask = tf.placeholder("float", [None, n_output])

        sensory_inputs, rule_inputs = tf.split(
            self.x, [hp['rule_start'], hp['n_rule']], axis=-1)

        sensory_rnn_inputs = tf.layers.dense(sensory_inputs, n_rnn, name='sen_input')

        if 'mix_rule' in hp and hp['mix_rule'] is True:
            # rotate rule matrix
            kernel_initializer = tf.orthogonal_initializer()
            rule_inputs = tf.layers.dense(
                rule_inputs, hp['n_rule'], name='mix_rule',
                use_bias=False, trainable=False,
                kernel_initializer=kernel_initializer)

        rule_rnn_inputs = tf.layers.dense(rule_inputs, n_rnn, name='rule_input', use_bias=False)

        rnn_inputs = sensory_rnn_inputs + rule_rnn_inputs

        # Recurrent activity
        cell = LeakyRNNCellSeparateInput(
            n_rnn, hp['alpha'],
            sigma_rec=hp['sigma_rec'],
            activation=hp['activation'],
            w_rec_init=hp['w_rec_init'],
            rng=self.rng)

        # Dynamic rnn with time major
        self.h, states = rnn.dynamic_rnn(
            cell, rnn_inputs, dtype=tf.float32, time_major=True)

        # Output
        h_shaped = tf.reshape(self.h, (-1, n_rnn))
        y_shaped = tf.reshape(self.y, (-1, n_output))
        # y_hat shape (n_time*n_batch, n_unit)
        y_hat = tf.layers.dense(
            h_shaped, n_output, activation=tf.nn.sigmoid, name='output')
        # Least-square loss
        self.cost_lsq = tf.reduce_mean(
            tf.square((y_shaped - y_hat) * self.c_mask))

        self.y_hat = tf.reshape(y_hat,
                                (-1, tf.shape(self.h)[1], n_output))
        y_hat_fix, y_hat_ring = tf.split(
            self.y_hat, [1, n_output - 1], axis=-1)
        self.y_hat_loc = tf_popvec(y_hat_ring)

    def _set_weights_separate(self, hp):
        """Set model attributes for several weight variables."""
        n_input = hp['n_input']
        n_rnn = hp['n_rnn']
        n_output = hp['n_output']

        for v in self.var_list:
            if 'rnn' in v.name:
                if 'kernel' in v.name or 'weight' in v.name:
                    self.w_rec = v
                else:
                    self.b_rec = v
            elif 'sen_input' in v.name:
                if 'kernel' in v.name or 'weight' in v.name:
                    self.w_sen_in = v
                else:
                    self.b_in = v
            elif 'rule_input' in v.name:
                self.w_rule = v
            else:
                assert 'output' in v.name
                if 'kernel' in v.name or 'weight' in v.name:
                    self.w_out = v
                else:
                    self.b_out = v

        # check if the recurrent and output connection has the correct shape
        if self.w_out.shape != (n_rnn, n_output):
            raise ValueError('Shape of w_out should be ' +
                             str((n_rnn, n_output)) + ', but found ' +
                             str(self.w_out.shape))
        if self.w_rec.shape != (n_rnn, n_rnn):
            raise ValueError('Shape of w_rec should be ' +
                             str((n_rnn, n_rnn)) + ', but found ' +
                             str(self.w_rec.shape))
        if self.w_sen_in.shape != (hp['rule_start'], n_rnn):
            raise ValueError('Shape of w_sen_in should be ' +
                             str((hp['rule_start'], n_rnn)) + ', but found ' +
                             str(self.w_sen_in.shape))
        if self.w_rule.shape != (hp['n_rule'], n_rnn):
            raise ValueError('Shape of w_in should be ' +
                             str((hp['n_rule'], n_rnn)) + ', but found ' +
                             str(self.w_rule.shape))

    def initialize(self):
        """Initialize the model for training."""
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())

    def restore(self, load_dir=None):
        """restore the model"""
        sess = tf.get_default_session()
        if load_dir is None:
            load_dir = self.model_dir
        save_path = os.path.join(load_dir, 'model.ckpt')
        try:
            self.saver.restore(sess, save_path)
        except:
            # Some earlier checkpoints only stored trainable variables
            self.saver = tf.train.Saver(self.var_list)
            self.saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

    def save(self):
        """Save the model."""
        sess = tf.get_default_session()
        save_path = os.path.join(self.model_dir, 'model.ckpt')
        self.saver.save(sess, save_path)
        print("Model saved in file: %s" % save_path)

    def set_optimizer(self, extra_cost=None, var_list=None):
        """Recompute the optimizer to reflect the latest cost function.

        This is useful when the cost function is modified throughout training

        Args:
            extra_cost : tensorflow variable,
            added to the lsq and regularization cost
        """
        cost = self.cost_lsq + self.cost_reg
        if extra_cost is not None:
            cost += extra_cost

        if var_list is None:
            var_list = self.var_list

        self.grads_and_vars = self.opt.compute_gradients(cost, var_list)
        # gradient clipping
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var)
                      for grad, var in self.grads_and_vars]
        self.train_step = self.opt.apply_gradients(capped_gvs)

    def lesion_units(self, sess, units, verbose=False):
        """Lesion units given by units

        Args:
            sess: tensorflow session
            units : can be None, an integer index, or a list of integer indices
        """

        # Convert to numpy array
        if units is None:
            return
        elif not hasattr(units, '__iter__'):
            units = np.array([units])
        else:
            units = np.array(units)

        # This lesioning will work for both RNN and GRU
        n_input = self.hp['n_input']
        for v in self.var_list:
            
            if 'kernel' in v.name or 'weight' in v.name:
                # Connection weights
                v_val = sess.run(v)
                if 'output' in v.name:
                    # output weights
                    v_val[units, :] = 0
                elif 'rnn' in v.name:
                    # recurrent weights
                    v_val[n_input + units, :] = 0
                sess.run(v.assign(v_val))

        if verbose:
            print('Lesioned units:')
            print(units)

    def set_unit_activations(self, sess, units, activations, verbose=False):
        """Set activations of given units to the values given by activations

        Args:
            sess: tensorflow session
            units : can be None, an integer index, or a list of integer indices
        """

        # Convert to numpy array
        if units is None:
            return
        elif not hasattr(units, '__iter__'):
            units = np.array([units])
        else:
            units = np.array(units)

        # This lesioning will work for both RNN and GRU
        n_input = self.hp['n_input']
        for v in self.var_list:
        
            v_val = sess.run(v)
            if 'kernel' in v.name or 'weight' in v.name:
                # Connection weights
                
                if 'output' in v.name:
                    # output weights
                    #v_val[units, :] = activations
                    continue
                elif 'rnn' in v.name:
                    # recurrent weights
                    v_val[n_input + units, :] = 0

                sess.run(v.assign(v_val))

            elif 'bias' in v.name and 'rnn' in v.name:

                v_val[units] = activations

                sess.run(v.assign(v_val))

def new_create_recurrent_weights_ring_attractor(N, g, rho, Si):

    R = g * np.random.normal(0, np.sqrt(1./(N)), (N,N))

    y1 = get_gaussian_vector( 0, 1, N) # Unit vectors required for the rank-two structure (see Methods)
    y2 = get_gaussian_vector( 0, 1, N)

    x1 = get_gaussian_vector( 0, 1, N)
    x2 = get_gaussian_vector( 0, 1, N)
    x3 = get_gaussian_vector( 0, 1, N)
    x4 = get_gaussian_vector( 0, 1, N)

    m1 = np.sqrt(Si**2 - rho**2)*x1 + rho*y1
    m2 = np.sqrt(Si**2 - rho**2)*x2 + rho*y2
    n1 = np.sqrt(Si**2 - rho**2)*x3 + rho*y1
    n2 = np.sqrt(Si**2 - rho**2)*x4 + rho*y2

    M = (np.outer( m1 , n1 ) + np.outer( m2 , n2 )) / N
    J = M + R

    return J

def new_create_recurrent_weights_plane_attractor(N, g, rho, Si):
    R = g * np.random.normal(0, np.sqrt(1./N), (N, N))

    # Gaussian basis vectors
    y1, y2 = get_gaussian_vector(0, 1, N), get_gaussian_vector(0, 1, N)
    y3, y4 = get_gaussian_vector(0, 1, N), get_gaussian_vector(0, 1, N)

    x1, x2 = get_gaussian_vector(0, 1, N), get_gaussian_vector(0, 1, N)
    x3, x4 = get_gaussian_vector(0, 1, N), get_gaussian_vector(0, 1, N)
    x5, x6 = get_gaussian_vector(0, 1, N), get_gaussian_vector(0, 1, N)
    x7, x8 = get_gaussian_vector(0, 1, N), get_gaussian_vector(0, 1, N)

    # Structured vectors with shared correlations
    m1 = np.sqrt(Si**2 - rho**2) * x1 + rho * y1
    m2 = np.sqrt(Si**2 - rho**2) * x2 + rho * y2
    n1 = np.sqrt(Si**2 - rho**2) * x3 + rho * y1
    n2 = np.sqrt(Si**2 - rho**2) * x4 + rho * y2

    m3 = np.sqrt(Si**2 - rho**2) * x5 + rho * y3
    m4 = np.sqrt(Si**2 - rho**2) * x6 + rho * y4
    n3 = np.sqrt(Si**2 - rho**2) * x7 + rho * y3
    n4 = np.sqrt(Si**2 - rho**2) * x8 + rho * y4

    # Rank-4 structured component (to allow a more flexible 2D attractor)
    M = (np.outer(m1, n1) + np.outer(m2, n2) + np.outer(m3, n3) + np.outer(m4, n4)) / N

    # Final recurrent matrix
    J = M + R
    return J


def new_create_recurrent_weights_two_ring_attractor(N, g, rho, Si):

    R = g * np.random.normal(0, np.sqrt(1./(N)), (N,N))

    y1 = get_gaussian_vector( 0, 1, N) # Unit vectors required for the rank-two structure (see Methods)
    y2 = get_gaussian_vector( 0, 1, N)

    x1 = get_gaussian_vector( 0, 1, N)
    x2 = get_gaussian_vector( 0, 1, N)
    x3 = get_gaussian_vector( 0, 1, N)
    x4 = get_gaussian_vector( 0, 1, N)

    m1 = np.sqrt(Si**2 - rho**2)*x1 + rho*y1
    m2 = np.sqrt(Si**2 - rho**2)*x2 + rho*y2
    n1 = np.sqrt(Si**2 - rho**2)*x3 + rho*y1
    n2 = np.sqrt(Si**2 - rho**2)*x4 + rho*y2

    M1 = (np.outer( m1 , n1 ) + np.outer( m2 , n2 )) / N

    y1 = get_gaussian_vector( 0, 1, N) # Unit vectors required for the rank-two structure (see Methods)
    y2 = get_gaussian_vector( 0, 1, N)

    x1 = get_gaussian_vector( 0, 1, N)
    x2 = get_gaussian_vector( 0, 1, N)
    x3 = get_gaussian_vector( 0, 1, N)
    x4 = get_gaussian_vector( 0, 1, N)

    m1 = np.sqrt(Si**2 - rho**2)*x1 + rho*y1
    m2 = np.sqrt(Si**2 - rho**2)*x2 + rho*y2
    n1 = np.sqrt(Si**2 - rho**2)*x3 + rho*y1
    n2 = np.sqrt(Si**2 - rho**2)*x4 + rho*y2

    M2 = (np.outer( m1 , n1 ) + np.outer( m2 , n2 )) / N

    J = M1 + M2 + R

    return J

def create_recurrent_weights_ring_attractor(N, g, structure_strength):
    """
    Creates the recurrent weight matrix J for an RNN, designed to have a ring attractor,
    following Mastrogiuseppe and Ostojic (2018).  This implementation
    creates a rank-2 connectivity structure suitable for generating a ring attractor.

    Args:
        N (int): Number of neurons in the network.
        g (float): Strength of the random component of the connectivity.
        structure_strength (float): Strength of the low-rank structure.
            Corresponds to m^T n / N in the paper.  This should be the
            SAME for both sets of m and n vectors.

    Returns:
        numpy.ndarray: The recurrent weight matrix J (N x N).
    """

    # 1. Define the left and right connectivity vectors, m1, n1, m2, and n2.
    #    These vectors determine the low-rank structure in the connectivity.
    #    For a ring attractor, the overlap between m1 and n1 should be the
    #    same as the overlap between m2 and n2.
    m1 = np.zeros(N)
    n1 = np.zeros(N)
    m2 = np.zeros(N)
    n2 = np.zeros(N)

    # Create oscillatory patterns, similar to the previous implementation,
    # but now we have two sets of m and n.
    for i in range(N):
        m1[i] = np.cos(2 * np.pi * i / N)
        n1[i] = np.sin(2 * np.pi * i / N)
        m2[i] = np.cos(2 * np.pi * i / N + np.pi / 4)  # Shifted phase for m2
        n2[i] = np.sin(2 * np.pi * i / N + np.pi / 4)  # Shifted phase for n2

    # 2. Calculate the overlaps (m^T n / N) - Enforce equal overlaps.
    overlap1 = np.dot(m1, n1) / N
    overlap2 = np.dot(m2, n2) / N

    # Use the provided structure_strength, or calculate it from the overlaps.
    if structure_strength is None:
      structure_strength = overlap1 # or overlap2, they should be equal

    # 3. Create the low-rank component, P.  Now it's a rank-2 structure.
    P = (structure_strength / N) * (np.outer(m1, n1) + np.outer(m2, n2))

    # 4. Create the random component, gx.
    x = np.random.randn(N, N) * (g / np.sqrt(N))

    # 5. Combine the low-rank and random components.
    J = g * x + P

    return J

def init_bernoulli_lowrank_plus_noise(N, p=10, g=0.5):
    # Generate p Bernoulli patterns (0 or 1)
    Z = np.random.binomial(1, 0.5, size=(N, p))

    # Low-rank part: outer product sum
    W_lowrank = (Z @ Z.T) / N

    # Full-rank random noise matrix J
    J = np.random.normal(0, g**2 / N, size=(N, N))

    # Combine
    W = W_lowrank + J

    # Set autapses
    np.fill_diagonal(W, 1.0)

    return W

def get_gaussian_vector(mean, std, N):

	if std>0:
		return np.random.normal (mean, std, N )
	else:
		return mean*np.ones(N)