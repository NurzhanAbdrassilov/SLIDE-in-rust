use std::f64::consts::PI;
use std::sync::{Arc, Mutex};
use std::cell::RefCell;
use rand::Rng;
use crate::psl_array::PSLArray;
use rand::SeedableRng;
use rand::rngs::StdRng;
use crate::config::ADAM;

lazy_static::lazy_static! {
    static ref BM_CACHE: Mutex<(f64, bool)> = Mutex::new((0.0, false));
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NodeType {
    ReLU,
    Softmax,
}

#[derive(Clone, Debug, Default)]
pub struct Train {
    pub last_delta_for_bp: f32,
    pub last_activation: f32,
    pub last_gradient: f32,
    pub active_input_ids: i32,
}

pub fn generate_normal_random(mean: f64, stddev: f64, rng: &mut StdRng) -> f64 {
    let mut cache = BM_CACHE.lock().unwrap();
    if !cache.1 {
        let u1: f64 = rng.gen();
        let u2: f64 = rng.gen();
        let sqrtlog = (-2.0 * u1.ln()).sqrt();
        let z0 = sqrtlog * (2.0 * PI * u2).cos();
        let z1 = sqrtlog * (2.0 * PI * u2).sin();
        cache.0 = z1;
        cache.1 = true;
        z0 * stddev + mean
    } else {
        cache.1 = false;
        cache.0 * stddev + mean
    }
}

pub struct Node {
    pub dim: usize,
    pub id_in_layer: usize,
    pub layer_num: usize,
    pub current_batch_size: usize,
    pub train: Vec<Train>,
    pub node_type: NodeType,

    pub no_of_nodes: usize,
    pub offset: usize,
    pub indices_in_tables: Option<Vec<usize>>,
    pub indices_in_buckets: Option<Vec<usize>>,
    pub weights: Option<Arc<PSLArray<f32>>>,
    pub bias: Option<Arc<PSLArray<f32>>>,
    pub weights_test: Option<Arc<RefCell<Vec<f32>>>>,
    pub mirror_weights: Option<Vec<f32>>,
    pub adam_avg_mom: Option<Vec<f32>>,
    pub adam_avg_vel: Option<Vec<f32>>,
    pub t: Option<Vec<f32>>,
    pub t_batch: Option<Arc<PSLArray<f32>>>,
    pub t_bias_batch: Option<Arc<PSLArray<f32>>>,
    pub t_batch_local: Option<Vec<Vec<f32>>>,
    pub t_bias_batch_local: Option<Vec<f32>>,
    pub update: Option<Vec<i32>>,
    pub tbias: f32,
    pub adam_avg_mom_bias: f32,
    pub adam_avg_vel_bias: f32,
    pub mirror_bias: f32,
    pub active_inputs: usize,
}

impl Default for Node {
    fn default() -> Self {
        Node {
            dim: 0,
            id_in_layer: 0,
            layer_num: 0,
            current_batch_size: 0,
            train: Vec::new(),
            node_type: NodeType::ReLU,

            no_of_nodes: 0,
            offset: 0,
            indices_in_tables: None,
            indices_in_buckets: None,
            weights: None,
            bias: None,
            weights_test: None,
            mirror_weights: None,
            adam_avg_mom: None,
            adam_avg_vel: None,
            t: None,
            t_batch: None,
            t_bias_batch: None,
            t_batch_local: None,
            t_bias_batch_local: None,
            update: None,
            tbias: 0.0,
            adam_avg_mom_bias: 0.0,
            adam_avg_vel_bias: 0.0,
            mirror_bias: 0.0,
            active_inputs: 0,
        }
    }
}

impl Node {
    #[allow(dead_code)]
    pub fn new(
        dim: usize,
        node_id: usize,
        layer_id: usize,
        node_type: NodeType,
        batch_size: usize,
        _weights: *const f32,
        _bias: f32,
        adam_avg_mom: Option<Vec<f32>>,
        adam_avg_vel: Option<Vec<f32>>,
        _ctx: i32, 
    ) -> Self {
        let mut node = Node {
            dim,
            id_in_layer: node_id,
            node_type,
            layer_num: layer_id,
            current_batch_size: batch_size,
            train: vec![Train::default(); batch_size],
            active_inputs: 0,

            indices_in_tables: None,
            indices_in_buckets: None,
            weights: None,
            bias: None,
            weights_test: None,
            mirror_weights: None,
            update: None,
            t_batch: None,
            t_bias_batch: None,
            t_batch_local: None,
            t_bias_batch_local: None,

            no_of_nodes: 0,
            offset: 0,
            tbias: 0.0,
            mirror_bias: 0.0,
            adam_avg_mom_bias: 0.0,
            adam_avg_vel_bias: 0.0,

            adam_avg_mom: None,
            adam_avg_vel: None,
            t: None,
        };

        if ADAM {
            node.adam_avg_mom = adam_avg_mom;
            node.adam_avg_vel = adam_avg_vel;
            node.t = Some(vec![0.0; dim]);
        }

        node
    }

    pub fn update(
    &mut self,
    no_of_nodes: usize,
    dim: usize,
    node_id: usize,
    layer_id: usize,
    node_type: NodeType,
    batch_size: usize,
    weights: Option<Arc<PSLArray<f32>>>,
    weight_offset: usize,
    bias: Option<Arc<PSLArray<f32>>>,
    adam_avg_mom: Option<Vec<f32>>,
    adam_avg_vel: Option<Vec<f32>>,
    train_blob: Vec<Train>,
    t_batch: Option<Arc<PSLArray<f32>>>,
    t_bias_batch: Option<Arc<PSLArray<f32>>>,
    weights_test: Option<Arc<RefCell<Vec<f32>>>>,
    ) {
        self.dim = dim;
        self.id_in_layer = node_id;
        self.layer_num = layer_id;
        self.node_type = node_type;
        self.current_batch_size = batch_size;

        if ADAM {
            self.adam_avg_mom = adam_avg_mom;
            self.adam_avg_vel = adam_avg_vel;
            self.t = Some(vec![0.0; dim]);
            self.t_batch = t_batch;
            self.t_bias_batch = t_bias_batch;
        }

        self.train = train_blob;

        self.active_inputs = 0;
        self.weights = weights;
        self.weights_test = weights_test;
        self.offset = weight_offset;
        self.bias = bias;
        self.no_of_nodes = no_of_nodes;
    }



    pub fn get_last_activation(&self, input_id: usize) -> f32 {
        if self.train[input_id].active_input_ids != 1 {
            0.0
        } else {
            self.train[input_id].last_activation
        }
    }

    /// Accumulates gradient delta for backpropagation
    /// Only processes gradients when the node is active and has positive activation
    pub fn increment_delta(&mut self, input_id: usize, increment_value: f32) {
        // Ensure input is marked as active for gradient computation
        if self.train[input_id].active_input_ids != 1 {
            self.train[input_id].active_input_ids = 1;
            self.active_inputs += 1;
        }
        
        // Only accumulate gradients for active neurons (ReLU activation > 0)
        if self.train[input_id].last_activation > 0.0 {
            self.train[input_id].last_delta_for_bp += increment_value;
        } else {
            // Skip gradient for inactive neurons to prevent infinite loops
            if self.layer_num == 0 && self.id_in_layer == 0 {
                return;
            }
        }
    }

    #[allow(dead_code)]
    pub fn get_input_active(&self, input_id: usize) -> bool {
        self.train[input_id].active_input_ids == 1
    }

    #[allow(dead_code)]
    pub fn get_active_inputs(&self) -> bool {
        self.active_inputs > 0
    }

    pub fn get_activation(
    &mut self,
    indices: &[usize],
    values: &[f32],
    length: usize,
    input_id: usize,
    local_weights_arr: &[f32],
    local_bias_arr: &[f32],
    ) -> f32 {
        assert!(input_id <= self.current_batch_size, "Input ID more than Batch Size");

        if self.train[input_id].active_input_ids != 1 {
            self.train[input_id].active_input_ids = 1;
            self.active_inputs += 1;
        }

        // Initialize activation value for computation
        if self.train[input_id].active_input_ids != 1 {
            self.train[input_id].last_activation = 0.0;
        }

        let weights_batch = self
            .weights
            .as_ref()
            .expect("weights must be initialized")
            .as_ref()
            .batch;

        let base = (self.id_in_layer * self.dim) % weights_batch;
        let local_weights = &local_weights_arr[base..];

        // Compute weighted sum of inputs
        for i in 0..length {
            let idx = indices[i];
            self.train[input_id].last_activation += local_weights[idx] * values[i];
        }

        // Add bias term
        let bias_batch = self
            .bias
            .as_ref()
            .expect("bias must be initialized")
            .as_ref()
            .batch;

        let bias_idx = self.id_in_layer % bias_batch;
        let bias_val = local_bias_arr[bias_idx];
        self.train[input_id].last_activation += bias_val;

        match self.node_type {
            NodeType::ReLU => {
                if self.train[input_id].last_activation < 0.0 {
                    self.train[input_id].last_activation = 0.0;
                    self.train[input_id].last_gradient = 0.0; 
                    self.train[input_id].last_delta_for_bp = 0.0;
                } else {
                    self.train[input_id].last_gradient = 1.0;
                }
            }
            NodeType::Softmax => {}
        }

        self.train[input_id].last_activation
    }

    /// Computes softmax probabilities and cross-entropy loss gradients
    pub fn compute_extra_stats_for_softmax(&mut self,
                                           normalization_constant: f32,
                                           input_id: usize,
                                           label: &[usize]) {
        // Ensure input is marked as active
        if self.train[input_id].active_input_ids != 1 {
            self.train[input_id].active_input_ids = 1;
            self.active_inputs += 1;
        }
        
        // Apply softmax normalization
        let scaled = self.train[input_id].last_activation / (normalization_constant + 1e-7);
        self.train[input_id].last_activation = scaled;
        self.train[input_id].last_gradient = 1.0;
        
        // Compute cross-entropy loss gradient
        if label.contains(&self.id_in_layer) {
            self.train[input_id].last_delta_for_bp =
                (1.0 / label.len() as f32 - scaled) / self.current_batch_size as f32;
        } else {
            self.train[input_id].last_delta_for_bp = (-scaled) / self.current_batch_size as f32;
        }
    }

    /// Performs backpropagation to update weights and propagate gradients
    pub fn back_propagate(&mut self,
                          previous_nodes: &mut [Node],
                          prev_active_ids: &[usize],
                          prev_active_size: usize,
                          learning_rate: f32,
                          input_id: usize,
                          local_weights: &[f32]) {
        // Ensure input is marked as active
        if self.train[input_id].active_input_ids != 1 {
            self.train[input_id].active_input_ids = 1;
            self.active_inputs += 1;
        }
        
        let delta = self.train[input_id].last_delta_for_bp;
        
        // Update weights and propagate gradients to previous layer
        for &prev_id in prev_active_ids.iter().take(prev_active_size) {
            let grad_t = delta * previous_nodes[prev_id].train[input_id].last_activation;
            
            // Accumulate weight gradients
            if ADAM {
                if let Some(ref mut t) = self.t {
                    t[prev_id] += grad_t;
                }
            } else if let Some(ref mut mirror) = self.mirror_weights {
                mirror[prev_id] += learning_rate * grad_t;
            }
            
            // Propagate gradient to previous layer
            previous_nodes[prev_id].increment_delta(input_id, delta * local_weights[prev_id]);
        }
        
        // Update bias
        if ADAM {
            self.tbias += delta;
        } else {
            self.mirror_bias += learning_rate * delta;
        }
        
        // Reset for next iteration
        self.train[input_id].active_input_ids = 0;
        self.train[input_id].last_delta_for_bp = 0.0;
        self.active_inputs -= 1;
    }

    /// Performs backpropagation for the first layer (input layer)
    pub fn back_propagate_first_layer(&mut self,
                                      nnz_indices: &[usize],
                                      nnz_values: &[f32],
                                      nnz_size: usize,
                                      learning_rate: f32,
                                      input_id: usize) {
        // Ensure input is marked as active
        if self.train[input_id].active_input_ids != 1 {
            self.train[input_id].active_input_ids = 1;
            self.active_inputs += 1;
        }
        
        let delta = self.train[input_id].last_delta_for_bp;
        
        // Update weights for sparse input features
        for i in 0..nnz_size {
            let idx = nnz_indices[i];
            let grad_t = delta * nnz_values[i];
            
            if ADAM {
                if let Some(ref mut t) = self.t {
                    t[idx] += grad_t;
                }
            } else if let Some(ref mut mirror) = self.mirror_weights {
                mirror[idx] += learning_rate * grad_t;
            }
        }
        
        // Update bias
        if ADAM {
            self.tbias += delta;
        } else {
            self.mirror_bias += learning_rate * delta;
        }
        
        // Reset for next iteration
        self.train[input_id].active_input_ids = 0;
        self.train[input_id].last_delta_for_bp = 0.0;
        self.active_inputs -= 1;
    }

    pub fn set_last_activation(&mut self, input_id: usize, real_activation: f32) {
        self.train[input_id].last_activation = real_activation;
    }

    #[allow(dead_code)]
    pub fn perturb_weight(&mut self, weight_id: usize, delta: f32) -> f32 {
        if let Some(ref mut mirror) = self.mirror_weights {
            mirror[weight_id] += delta;
            mirror[weight_id]
        } else {
            0.0
        }
    }

    #[allow(dead_code)]
    pub fn get_gradient(&self, _weight_id: usize, input_id: usize, input_val: f32) -> f32 {
        -self.train[input_id].last_delta_for_bp * input_val
    }
}


impl Drop for Node {
    fn drop(&mut self) {
    }
}


