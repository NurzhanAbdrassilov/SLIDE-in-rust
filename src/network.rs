//! Neural Network implementation for SLIDE algorithm
//! Handles sparse neural network training with distributed weight storage

use std::sync::Arc;
use std::time::Instant;

use crate::config::*;
use crate::layer::Layer;
use crate::node::NodeType;
use crate::psl_array::CacheEntry;

pub struct Network {
    pub hiddenlayers: Vec<Layer>,
    learning_rate: f32,
    number_of_layers: usize,
    sizes_of_layers: Vec<usize>,
    layers_types: Vec<NodeType>,
    sparsity: Vec<f32>,
    current_batch_size: usize,
    worker_id: usize,
}

impl Network {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sizes_of_layers: Vec<usize>,
        layers_types: Vec<NodeType>,
        number_of_layers: usize,
        batch_size: usize,
        lr: f32,
        input_dim: usize,
        k: &[usize],
        l: &[usize],
        range_pow: &[usize],
        sparsity: Vec<f32>,
        _npz_placeholder: (),
        worker_id: usize,
    ) -> Self {
        let mut hiddenlayers = Vec::with_capacity(number_of_layers);

        for i in 0..number_of_layers {
            let prev_dim = if i == 0 { input_dim } else { sizes_of_layers[i - 1] };
            let curr_dim = sizes_of_layers[i];

            hiddenlayers.push(Layer::new(
                worker_id,
                curr_dim,
                prev_dim,
                i,
                layers_types[i],
                batch_size,
                k[i],
                l[i],
                range_pow[i],
                sparsity[i],
                None,
                None,
                None,
                None,
            ));
        }

        let mut net = Network {
            hiddenlayers,
            learning_rate: lr,
            number_of_layers,
            sizes_of_layers,
            layers_types,
            sparsity,
            current_batch_size: batch_size,
            worker_id,
        };
        net.prefetch_weights_and_biases();
        net
    }

    pub fn get_layer(&mut self, layer_id: usize) -> &mut Layer {
        assert!(layer_id < self.number_of_layers, "LayerID out of bounds");
        &mut self.hiddenlayers[layer_id]
    }

    /// Reports network weight statistics for monitoring training progress
    pub fn report_weight_stats(&self) {
        println!("Network Weight Statistics:");
        for (layer_idx, layer) in self.hiddenlayers.iter().enumerate() {
            println!("  Layer {}: {} nodes", layer_idx, layer.nodes.len());
            
            // Sample first few nodes to monitor weight updates
            for (node_idx, node) in layer.nodes.iter().enumerate().take(3) {
                if let Some(ref weights) = node.mirror_weights {
                    let sum: f32 = weights.iter().sum();
                    let avg = sum / weights.len() as f32;
                    let max_val = weights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let min_val = weights.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                    println!("    Node {}: weights len={}, avg={:.6}, range=[{:.6}, {:.6}]", 
                            node_idx, weights.len(), avg, min_val, max_val);
                }
                
                if let Some(ref t_weights) = node.t {
                    let sum: f32 = t_weights.iter().sum();
                    let avg = sum / t_weights.len() as f32;
                    println!("    Node {} gradients: len={}, avg={:.6}", 
                            node_idx, t_weights.len(), avg);
                }
                
                if node_idx >= 2 { break; } // Sample first 3 nodes per layer
            }
        }
    }

    pub fn predict_class(
        &mut self,
        input_indices: &[Vec<usize>],
        input_values: &[Vec<f32>],
        lengths: &[usize],
        labels: &[Vec<usize>],
        label_sizes: &[usize],
        count: usize,
    ) -> i32 {
        let mut correct_pred = 0i32;
        let t1 = Instant::now();

        let chunk = count / NUM_WAIT as usize;
        let mut start = self.worker_id * chunk;
        let mut end = start + chunk;
        if self.worker_id == (NUM_WAIT as usize - 1) {
            end = count;
        }

        debug_assert!(self.sparsity.len() >= self.number_of_layers * 2);

        for i in start..end {
            let mut active_nodes_per_layer = vec![Vec::<usize>::new(); self.number_of_layers + 1];
            let mut active_values_per_layer = vec![Vec::<f32>::new(); self.number_of_layers + 1];
            let mut sizes = vec![0usize; self.number_of_layers + 1];

            active_nodes_per_layer[0] = input_indices[i].clone();
            active_values_per_layer[0] = input_values[i].clone();
            sizes[0] = lengths[i];

            for j in 0..self.number_of_layers {
                let (fw, fb) = {
                    let layer_ref = &self.hiddenlayers[j];
                    (layer_ref.full_weights.clone(), layer_ref.full_bias.clone())
                };

                let _ = self.hiddenlayers[j].query_active_node_and_compute_activations(
                    &mut active_nodes_per_layer,
                    &mut active_values_per_layer,
                    &mut sizes,
                    j,
                    i,
                    &labels[i],
                    self.sparsity[self.number_of_layers + j],
                    -1,
                    &fw,
                    &fb,
                );
            }

            let no_of_classes = sizes[self.number_of_layers];
            let mut max_act = f32::NEG_INFINITY;
            let mut predict_class = usize::MAX;

            for k in 0..no_of_classes {
                let node_id = active_nodes_per_layer[self.number_of_layers][k];
                let cur_act = self.hiddenlayers[self.number_of_layers - 1]
                    .get_node_by_id(node_id)
                    .get_last_activation(i);
                if cur_act > max_act {
                    max_act = cur_act;
                    predict_class = node_id;
                }
            }

            if labels[i][..label_sizes[i]].contains(&predict_class) {
                correct_pred += 1;
            }
        }

        let ms = (Instant::now() - t1).as_micros() as f64 / 1000.0;
        if ms > 10.0 { // Only report if inference is slow
            println!("Inference: {:.1}ms", ms);
        }
        correct_pred
    }

    pub fn prefetch_weights_and_biases(&mut self) {
        for l in 0..self.number_of_layers {
            for w in 0..(NUM_WAIT as usize) {
                self.hiddenlayers[l].full_bias[w] = if w == self.worker_id {
                    self.hiddenlayers[l].private_bias.clone()
                } else {
                    self.hiddenlayers[l].bias.get_arr_batch_idx(w)
                };
            }
            for w in 0..(NUM_WAIT as usize) {
                self.hiddenlayers[l].full_weights[w] = if w == self.worker_id {
                    self.hiddenlayers[l].private_weights.clone()
                } else {
                    self.hiddenlayers[l].weights.get_arr_batch_idx(w)
                };
            }
        }
    }

    pub fn clear_weights_and_biases(&mut self, keep_mine: bool) {
        for l in 0..self.number_of_layers {
            for w in 0..(NUM_WAIT as usize) {
                if keep_mine && w == self.worker_id {
                    continue;
                }
                self.hiddenlayers[l].full_bias[w] = Arc::new(CacheEntry {
                    data: std::sync::Mutex::new(Vec::new()),
                    key: String::new(),
                    batch: 0,
                    write_on_drop: false,
                });
            }
            for w in 0..(NUM_WAIT as usize) {
                if keep_mine && w == self.worker_id {
                    continue;
                }
                self.hiddenlayers[l].full_weights[w] = Arc::new(CacheEntry {
                    data: std::sync::Mutex::new(Vec::new()),
                    key: String::new(),
                    batch: 0,
                    write_on_drop: false,
                });
            }
            self.hiddenlayers[l].weights.flush();
            self.hiddenlayers[l].bias.flush();
        }
    }

    pub fn process_input(
        &mut self,
        input_indices: &[Vec<usize>],
        input_values: &[Vec<f32>],
        lengths: &[usize],
        labels: &[Vec<usize>],
        label_sizes: &[usize],
        iter: i32,
        rehash: bool,
        rebuild: bool,
    ) -> f32 {
        let logloss = 0.0f32;
        let mut avg_retrieval = vec![0usize; self.number_of_layers];

        if iter % 6946 == 6945 && self.number_of_layers > 1 {
            self.hiddenlayers[1].update_random_nodes();
        }

        let mut tmplr = self.learning_rate;
        if ADAM {
            let t = (iter + 1) as f32;
            tmplr = self.learning_rate * ((1.0 - BETA2.powf(t)).sqrt() / (1.0 - BETA1.powf(t)));
        }

        let chunk = self.current_batch_size / NUM_WAIT as usize;
        let mut start = self.worker_id * chunk;
        let mut end = start + chunk;
        if self.worker_id == (NUM_WAIT as usize - 1) {
            end = self.current_batch_size;
        }

        for i in start..end {
            let mut active_nodes_per_layer = vec![Vec::<usize>::new(); self.number_of_layers + 1];
            let mut active_values_per_layer = vec![Vec::<f32>::new(); self.number_of_layers + 1];
            let mut sizes = vec![0usize; self.number_of_layers + 1];

            active_nodes_per_layer[0] = input_indices[i].clone();
            active_values_per_layer[0] = input_values[i].clone();
            sizes[0] = lengths[i];

            for j in 0..self.number_of_layers {
                let (fw, fb) = {
                    let layer_ref = &self.hiddenlayers[j];
                    (layer_ref.full_weights.clone(), layer_ref.full_bias.clone())
                };

                let _ = self.hiddenlayers[j].query_active_node_and_compute_activations(
                    &mut active_nodes_per_layer,
                    &mut active_values_per_layer,
                    &mut sizes,
                    j,
                    i,
                    &labels[i],
                    self.sparsity[if cfg!(feature = "predict") { self.number_of_layers + j } else { j }],
                    -1,
                    &fw,
                    &fb,
                );
            }


            for j in (0..self.number_of_layers).rev() {
                let (left, right) = self.hiddenlayers.split_at_mut(j);
                let (cur_layer, _tail) = right.split_first_mut().unwrap();

                let is_last = j == self.number_of_layers - 1;

                for k in 0..sizes[j + 1] {
                    let node_id = active_nodes_per_layer[j + 1][k];
                    
                    // Ensure node_id is within bounds for this layer
                    let node_id = node_id.min(cur_layer.no_of_nodes.saturating_sub(1));

                    let norm_const = if is_last {
                        Some(cur_layer.get_normalization_constant(i))
                    } else {
                        None
                    };
                    let prev_dim        = cur_layer.previous_layer_num_of_nodes;
                    let weights_batch   = cur_layer.weights.batch;
                    let full_weights    = cur_layer.full_weights.clone();

                    let node = cur_layer.get_node_by_id(node_id);

                    if let Some(nc) = norm_const {
                        node.compute_extra_stats_for_softmax(nc, i, &labels[i][..label_sizes[i]]);
                    }

                    if j != 0 {
                        let prev_layer = &mut left[j - 1];

                        let worker_id = (node_id * prev_dim) / weights_batch;
                        let w_guard   = full_weights[worker_id].data.lock().unwrap();
                        let lw: &[f32] = &w_guard[..];

                        node.back_propagate(
                            &mut prev_layer.nodes[..],
                            &active_nodes_per_layer[j],
                            sizes[j],
                            tmplr,
                            i,
                            lw,
                        );
                    } else {
                        node.back_propagate_first_layer(
                            &input_indices[i],
                            &input_values[i],
                            lengths[i],
                            tmplr,
                            i,
                        );
                    }
                }

            }
        }

        if ADAM {
            for l in 0..self.number_of_layers {
                let dim = self.hiddenlayers[l].previous_layer_num_of_nodes;

                if let (Some(tb_arc), Some(tbb_arc)) =
                    (self.hiddenlayers[l].private_t_batch.clone(), self.hiddenlayers[l].private_t_bias.clone())
                {
                    {
                        let mut guard = tb_arc.data.lock().unwrap();
                        for m in 0..self.hiddenlayers[l].no_of_nodes {
                            let node = self.hiddenlayers[l].get_node_by_id(m);
                            let tvec = node.t.as_ref().expect("t present in ADAM");
                            let base = m * dim;
                            guard[base..base + dim].copy_from_slice(&tvec[..]);
                        }
                    }
                    {
                        let mut guard = tbb_arc.data.lock().unwrap();
                        for m in 0..self.hiddenlayers[l].no_of_nodes {
                            let node = self.hiddenlayers[l].get_node_by_id(m);
                            guard[m] = node.tbias;
                        }
                    }
                    self.hiddenlayers[l]
                        .t_batch
                        .as_ref()
                        .unwrap()
                        .flush_batch_idx(tb_arc, self.worker_id);
                    self.hiddenlayers[l]
                        .t_bias_batch
                        .as_ref()
                        .unwrap()
                        .flush_batch_idx(tbb_arc, self.worker_id);
                }
            }
            for w in 0..(NUM_WAIT as usize) {
                if w == self.worker_id { continue; }
                for l in 0..self.number_of_layers {
                    let dim = self.hiddenlayers[l].previous_layer_num_of_nodes;

                    let arr_t_batch = self.hiddenlayers[l].t_batch.as_ref().unwrap().get_arr_batch_idx(w);
                    let arr_t_bias  = self.hiddenlayers[l].t_bias_batch.as_ref().unwrap().get_arr_batch_idx(w);

                    let bias_range  = self.hiddenlayers[l].t_bias_batch.as_ref().unwrap()
                        .get_worker_range(w).expect("bias range");
                    let bias_off = bias_range.start as usize;

                    let batch_range = self.hiddenlayers[l].t_batch.as_ref().unwrap()
                        .get_worker_range(w).expect("t range");
                    let batch_off = batch_range.start as usize;

                    let start_n = self.hiddenlayers[l].bias.pt_start as usize;
                    let end_n   = self.hiddenlayers[l].bias.pt_end as usize;
                    {
                        let b_guard = arr_t_bias.data.lock().unwrap();
                        for i in start_n..end_n {
                            let node = self.hiddenlayers[l].get_node_by_id(i);
                            // Calculate the local offset within the worker's bias data
                            // The bias_off is the global start offset for this worker's data
                            if i >= bias_off && (i - bias_off) < b_guard.len() {
                                node.tbias += b_guard[i - bias_off];
                            }
                        }
                    }

                    let start = start_n * dim;
                    let end   = end_n * dim;
                    let w_guard = arr_t_batch.data.lock().unwrap();
                    let batch_start_node = batch_off / dim; // Convert batch offset to node-based offset
                    for idx in (start..end).step_by(dim) {
                        let node_id = idx / dim;
                        // Calculate the local offset within the worker's batch data
                        if node_id >= batch_start_node {
                            let local_node_idx = node_id - batch_start_node;
                            let local_idx = local_node_idx * dim;
                            if local_idx + dim <= w_guard.len() {
                                let src = &w_guard[local_idx..(local_idx + dim)];
                                let t_dst = self.hiddenlayers[l].get_node_by_id(node_id).t.as_mut().unwrap();
                                for d in 0..dim {
                                    t_dst[d] += src[d];
                                }
                            }
                        }
                    }

                    self.hiddenlayers[l].t_batch.as_ref().unwrap().flush();
                    self.hiddenlayers[l].t_bias_batch.as_ref().unwrap().flush();
                }
            }
        }

        for l in 0..self.number_of_layers {
            let do_rehash = rehash && self.sparsity[l] < 1.0;
            let do_rebuild = rebuild && self.sparsity[l] < 1.0;

            if do_rehash {
                self.hiddenlayers[l].hash_tables.clear();
            }
            if do_rebuild {
                self.hiddenlayers[l].update_table();
            }

            let dim = self.hiddenlayers[l].previous_layer_num_of_nodes;
            let chunk_n =
                (self.hiddenlayers[l].no_of_nodes + (NUM_WAIT as usize - 1)) / (NUM_WAIT as usize);
            let mut start_m = self.worker_id * chunk_n;
            let mut end_m = start_m + chunk_n;
            if self.worker_id == (NUM_WAIT as usize - 1) {
                end_m = self.hiddenlayers[l].no_of_nodes;
            }

            let local_tmp_weights_arr = self.hiddenlayers[l].private_weights.clone();
            let weights_offset = self.hiddenlayers[l].weights.pt_start as usize;
            let weights_size =
                (self.hiddenlayers[l].weights.pt_end - self.hiddenlayers[l].weights.pt_start)
                    as usize;

            let local_bias = self.hiddenlayers[l].private_bias.clone();
            let bias_offset = self.hiddenlayers[l].bias.pt_start as usize;

            debug_assert!(
                end_m * dim == self.hiddenlayers[l].weights.pt_end as usize
                    && start_m * dim == self.hiddenlayers[l].weights.pt_start as usize
            );

            for m in start_m..end_m {
                let node = self.hiddenlayers[l].get_node_by_id(m);
                let base = m * dim - weights_offset;
                debug_assert!(base < weights_size);

                let mut local_weights = vec![0.0f32; dim];
                {
                    let w_guard = local_tmp_weights_arr.data.lock().unwrap();
                    local_weights.copy_from_slice(&w_guard[base..base + dim]);
                }

                if ADAM {
                    for d in 0..dim {
                        let tgrad = node.t.as_ref().unwrap()[d];
                        let off   = m * dim + d;

                        let new_mom = {
                            let mom = node.adam_avg_mom.as_ref().unwrap()[off];
                            BETA1 * mom + (1.0 - BETA1) * tgrad
                        };
                        let new_vel = {
                            let vel = node.adam_avg_vel.as_ref().unwrap()[off];
                            BETA2 * vel + (1.0 - BETA2) * tgrad * tgrad
                        };

                        node.adam_avg_mom.as_mut().unwrap()[off] = new_mom;
                        node.adam_avg_vel.as_mut().unwrap()[off] = new_vel;
                        node.t.as_mut().unwrap()[d] = 0.0;

                        local_weights[d] += tmplr * new_mom / (new_vel.sqrt() + EPS);
                    }

                    node.adam_avg_mom_bias = BETA1 * node.adam_avg_mom_bias + (1.0 - BETA1) * node.tbias;
                    node.adam_avg_vel_bias = BETA2 * node.adam_avg_vel_bias + (1.0 - BETA2) * node.tbias * node.tbias;
                    let delta_bias = tmplr * node.adam_avg_mom_bias / (node.adam_avg_vel_bias.sqrt() + EPS);

                    {
                        let mut b_guard = local_bias.data.lock().unwrap();
                        b_guard[m - bias_offset] += delta_bias;
                    }
                    node.tbias = 0.0;

                    {
                        let mut w_guard = local_tmp_weights_arr.data.lock().unwrap();
                        w_guard[base..base + dim].copy_from_slice(&local_weights);
                    }
                } else {
                }
            }

            self.hiddenlayers[l]
                .weights
                .flush_batch_idx(local_tmp_weights_arr.clone(), self.worker_id);
            self.hiddenlayers[l]
                .bias
                .flush_batch_idx(local_bias.clone(), self.worker_id);

            if do_rehash {
                for m in start_m..end_m {
                    self.hiddenlayers[l].add_to_hash_table(dim, m);
                }
            }
        }


        self.clear_weights_and_biases(true);
        self.prefetch_weights_and_biases();

        for w in 0..(NUM_WAIT as usize) {
            if w == self.worker_id {
                continue;
            }
            for l in 0..self.number_of_layers {
                let range = self.hiddenlayers[l].weights.get_worker_range(w).expect("range");
                let range_start = range.start as usize;
                let range_end = range.end as usize;
                let dim = self.hiddenlayers[l].previous_layer_num_of_nodes;
                debug_assert!(((range_end - range_start) % dim) == 0);

                let arr = self.hiddenlayers[l].weights.get_arr_batch_idx(w);
                let guard = arr.data.lock().unwrap();
                for i in (range_start..range_end).step_by(dim) {
                    let m = i / dim;
                    let local_weights = &guard[i - range_start..i - range_start + dim];

                    self.hiddenlayers[l].add_to_hash_table(dim, m);
                }
            }
        }

        for l in 0..self.number_of_layers {
            let exclude_start = self.hiddenlayers[l].bias.pt_start as usize;
            let exclude_end = self.hiddenlayers[l].bias.pt_end as usize;

            for m in 0..exclude_start {
                if let Some(t) = self.hiddenlayers[l].get_node_by_id(m).t.as_mut() {
                    for v in t.iter_mut() {
                        *v = 0.0;
                    }
                }
                self.hiddenlayers[l].get_node_by_id(m).tbias = 0.0;
            }
            for m in exclude_end..self.hiddenlayers[l].no_of_nodes {
                if let Some(t) = self.hiddenlayers[l].get_node_by_id(m).t.as_mut() {
                    for v in t.iter_mut() {
                        *v = 0.0;
                    }
                }
                self.hiddenlayers[l].get_node_by_id(m).tbias = 0.0;
            }
        }

        // Optional: report retrieval statistics for debugging
        if rehash {
            let chunk = self.current_batch_size / NUM_WAIT as usize;
            let start = self.worker_id * chunk;
            let mut end = start + chunk;
            if self.worker_id == (NUM_WAIT as usize - 1) {
                end = self.current_batch_size;
            }
            if self.number_of_layers >= 2 {
                println!("Sample size: {:.1} {:.1}",
                    (avg_retrieval[0] as f32) / ((end - start) as f32),
                    (avg_retrieval[1] as f32) / (self.current_batch_size as f32)
                );
            }
        }

        logloss
    }

    pub fn save_weights(&self, file: &str) {
        for l in 0..self.number_of_layers {
            self.hiddenlayers[l].save_weights(file);
        }
    }
}
