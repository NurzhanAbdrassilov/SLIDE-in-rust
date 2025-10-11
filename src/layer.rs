use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use rand::{seq::SliceRandom, SeedableRng};
use rand::rngs::StdRng;
use rand::Rng;

use crate::config::*;
use crate::densified_min_hash::DensifiedMinhash;
use crate::densified_wta_hash::DensifiedWtaHash;
use crate::lsh::LSH;
use crate::node::{generate_normal_random, Node, NodeType, Train};
use crate::psl_array::{CacheEntry, PSLArray};
use crate::srp::SparseRandomProjection;
use crate::wta_hash::WtaHash;

pub struct Layer {
    pub layer_id: usize,
    pub no_of_nodes: usize,
    pub previous_layer_num_of_nodes: usize,
    pub no_of_active: usize,
    pub k: usize,
    pub l: usize,
    pub range_row: usize,
    pub batch_size: usize,
    pub worker_id: usize,

    pub node_type: NodeType,
    pub nodes: Vec<Node>,
    pub rand_node: Vec<usize>,
    pub normalization_constants: Option<Vec<f32>>,

    pub weights: Arc<PSLArray<f32>>,
    pub bias: Arc<PSLArray<f32>>,

    pub adam_avg_mom: Option<Vec<f32>>,
    pub adam_avg_vel: Option<Vec<f32>>,

    pub t_batch: Option<Arc<PSLArray<f32>>>,
    pub t_bias_batch: Option<Arc<PSLArray<f32>>>,

    pub private_weights: Arc<CacheEntry<f32>>,
    pub private_bias: Arc<CacheEntry<f32>>,
    pub private_t_batch: Option<Arc<CacheEntry<f32>>>,
    pub private_t_bias: Option<Arc<CacheEntry<f32>>>,

    pub full_weights: Vec<Arc<CacheEntry<f32>>>,
    pub full_bias: Vec<Arc<CacheEntry<f32>>>,

    pub train_array: Vec<Train>,

    pub hash_tables: LSH,
    pub wta_hasher: Option<WtaHash>,
    pub dwta_hasher: Option<DensifiedWtaHash>,
    pub min_hasher: Option<DensifiedMinhash>,
    pub srp: Option<SparseRandomProjection>,
    pub binids: Option<Vec<usize>>,
}

impl Layer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        worker_id: usize,
        no_of_nodes: usize,
        previous_layer_num_of_nodes: usize,
        layer_id: usize,
        node_type: NodeType,
        batch_size: usize,
        k: usize,
        l: usize,
        range_pow: usize,
        sparsity: f32,
        _weights_init: Option<Vec<f32>>,
        _bias_init: Option<Vec<f32>>,
        _adam_avg_mom_init: Option<Vec<f32>>,
        _adam_avg_vel_init: Option<Vec<f32>>,
    ) -> Self {
        let no_of_active = (no_of_nodes as f32 * sparsity).floor() as usize;

        let mut rand_node: Vec<usize> = (0..no_of_nodes).collect();
        let mut rng = StdRng::seed_from_u64(1234);
        rand_node.shuffle(&mut rng);

        let hash_tables = LSH::new(k, l, range_pow);

        let mut wta_hasher: Option<WtaHash> = None;
        let mut dwta_hasher: Option<DensifiedWtaHash> = None;
        let mut min_hasher: Option<DensifiedMinhash> = None;
        let mut srp: Option<SparseRandomProjection> = None;
        let mut binids: Option<Vec<usize>> = None;

        match HashFunction {
            1 => {
                wta_hasher = Some(WtaHash::new(k * l, previous_layer_num_of_nodes));
            }
            2 => {
                binids = Some(vec![0usize; previous_layer_num_of_nodes]);
                dwta_hasher = Some(DensifiedWtaHash::new(
                    (k * l),
                    previous_layer_num_of_nodes,
                ));
            }
            3 => {
                binids = Some(vec![0usize; previous_layer_num_of_nodes]);
                let mut mh = DensifiedMinhash::new(k * l, previous_layer_num_of_nodes);
                mh.get_map(previous_layer_num_of_nodes, binids.as_mut().unwrap());
                min_hasher = Some(mh);
            }
            4 => {
                srp = Some(SparseRandomProjection::new(
                    previous_layer_num_of_nodes,
                    k * l,
                    Ratio as usize,
                ));
            }
            _ => {}
        }

        let mut weights = Arc::new(PSLArray::with_workers(
            format!("layer{}_weights", layer_id),
            no_of_nodes * previous_layer_num_of_nodes,
            NUM_WAIT as usize,
            worker_id,
            previous_layer_num_of_nodes,
        ));
        let mut bias = Arc::new(PSLArray::with_workers(
            format!("layer{}_bias", layer_id),
            no_of_nodes,
            NUM_WAIT as usize,
            worker_id,
            1,
        ));

        let (adam_avg_mom, adam_avg_vel, t_batch, t_bias_batch): (
            Option<Vec<f32>>,
            Option<Vec<f32>>,
            Option<Arc<PSLArray<f32>>>,
            Option<Arc<PSLArray<f32>>>,
        ) = if ADAM {
            let tb = Arc::new(PSLArray::with_workers(
                format!("layer{}_t_batch", layer_id),
                (NUM_WAIT as usize) * no_of_nodes * previous_layer_num_of_nodes,
                NUM_WAIT as usize,
                worker_id,
                no_of_nodes * previous_layer_num_of_nodes,
            ));
            let tbb = Arc::new(PSLArray::with_workers(
                format!("layer{}_t_bias_batch", layer_id),
                (NUM_WAIT as usize) * no_of_nodes,
                NUM_WAIT as usize,
                worker_id,
                no_of_nodes,
            ));
            (
                Some(vec![0.0; no_of_nodes * previous_layer_num_of_nodes]),
                Some(vec![0.0; no_of_nodes * previous_layer_num_of_nodes]),
                Some(tb),
                Some(tbb),
            )
        } else {
            (None, None, None, None)
        };

        if !LOADWEIGHT {
            let mut rng = StdRng::seed_from_u64(33111);
            let mut weights_data = vec![0.0f32; no_of_nodes * previous_layer_num_of_nodes];
            let mut bias_data = vec![0.0f32; no_of_nodes];

            // Xavier/Glorot initialization for better stability
            let xavier_std = (2.0 / (previous_layer_num_of_nodes as f64 + no_of_nodes as f64)).sqrt();
            for w in &mut weights_data {
                *w = generate_normal_random(0.0, xavier_std, &mut rng) as f32;  // Use proper Xavier initialization
            }
            for b in &mut bias_data {
                *b = 0.0;  // Initialize biases to zero
            }

            // Initialize all worker partitions, not just the current one
            let w_mut = Arc::get_mut(&mut weights).expect("unique Arc for weights");
            
            // Initialize ALL partitions in the weights array, not just current worker's partition
            let mut wi = Cell::new(0usize);
            w_mut.init(|| {
                let i = wi.get();
                wi.set(i + 1);
                weights_data[i % weights_data.len()]
            });
            let b_mut = Arc::get_mut(&mut bias).expect("unique Arc for bias");
            
            // Initialize ALL partitions in the bias array
            let mut bi = Cell::new(0usize);
            b_mut.init(|| {
                let i = bi.get();
                bi.set(i + 1);
                bias_data[i % bias_data.len()]
            });
        }

        let full_weights: Vec<Arc<CacheEntry<f32>>> = (0..(NUM_WAIT as usize))
            .map(|_| {
                Arc::new(CacheEntry { data: std::sync::Mutex::new(Vec::new()), key: String::new(), batch: 0, write_on_drop: false })
            })
            .collect();
        let full_bias: Vec<Arc<CacheEntry<f32>>> = (0..(NUM_WAIT as usize))
            .map(|_| {
                Arc::new(CacheEntry { data: std::sync::Mutex::new(Vec::new()), key: String::new(), batch: 0, write_on_drop: false })
            })
            .collect();

        let private_weights = weights.get_arr_batch_idx(worker_id);
        let private_bias = bias.get_arr_batch_idx(worker_id);
        let private_t_batch = t_batch.as_ref().map(|tb| tb.get_arr_batch_idx(worker_id));
        let private_t_bias = t_bias_batch.as_ref().map(|tbb| tbb.get_arr_batch_idx(worker_id));

        let mut nodes: Vec<Node> = (0..no_of_nodes).map(|_| Node::default()).collect();
        let mut train_array: Vec<Train> = vec![Train::default(); no_of_nodes * batch_size];

        let t1 = Instant::now();
        for i in 0..no_of_nodes {
            // Calculate the slice of train_array for this node
            let node_train_start = i * batch_size;
            let node_train_end = (i + 1) * batch_size;
            let node_train_slice = train_array[node_train_start..node_train_end].to_vec();
            
            nodes[i].update(
                no_of_nodes,
                previous_layer_num_of_nodes,
                i,
                layer_id,
                node_type,
                batch_size,
                Some(weights.clone()),
                previous_layer_num_of_nodes * i,
                Some(bias.clone()),
                adam_avg_mom.clone(),
                adam_avg_vel.clone(),
                node_train_slice,  // Only the slice for this node
                t_batch.clone(),
                t_bias_batch.clone(),
                None,
            );
        }
        let _elapsed = t1.elapsed().as_micros();

        let normalization_constants = if matches!(node_type, NodeType::Softmax) {
            Some(vec![0.0; batch_size])
        } else {
            None
        };

        weights.flush();
        bias.flush();

        let mut layer = Layer {
            layer_id,
            no_of_nodes,
            previous_layer_num_of_nodes,
            no_of_active,
            k,
            l,
            range_row: range_pow,
            batch_size,
            worker_id,
            node_type,
            nodes,
            rand_node,
            normalization_constants,
            weights,
            bias,
            adam_avg_mom,
            adam_avg_vel,
            t_batch,
            t_bias_batch,
            private_weights,
            private_bias,
            private_t_batch,
            private_t_bias,
            full_weights,
            full_bias,
            train_array,
            hash_tables,
            wta_hasher,
            dwta_hasher,
            min_hasher,
            srp,
            binids,
        };
        let length = layer.previous_layer_num_of_nodes;
        for i in 0..layer.no_of_nodes {
            layer.add_to_hash_table(length, i);
        }
        return layer;
    }

    pub fn update_table(&mut self) {
        match HashFunction {
            1 => {
                self.wta_hasher = Some(WtaHash::new(self.k * self.l, self.previous_layer_num_of_nodes));
            }
            2 => {
                self.binids = Some(vec![0usize; self.previous_layer_num_of_nodes]);
                self.dwta_hasher = Some(DensifiedWtaHash::new(
                    (self.k * self.l),
                    self.previous_layer_num_of_nodes,
                ));
            }
            3 => {
                self.binids = Some(vec![0usize; self.previous_layer_num_of_nodes]);
                let mut mh = DensifiedMinhash::new(self.k * self.l, self.previous_layer_num_of_nodes);
                mh.get_map(self.previous_layer_num_of_nodes, self.binids.as_mut().unwrap());
                self.min_hasher = Some(mh);
            }
            4 => {
                self.srp = Some(SparseRandomProjection::new(
                    self.previous_layer_num_of_nodes,
                    self.k * self.l,
                    Ratio as usize,
                ));
            }
            _ => {}
        }
    }

    pub fn update_random_nodes(&mut self) {
        let mut rng = StdRng::seed_from_u64(1234);
        self.rand_node.shuffle(&mut rng);
    }

    pub fn update_weights(&mut self, weights_src: &[f32]) {
        let w = Arc::get_mut(&mut self.weights)
            .expect("update_weights called after sharing weights (Arc not unique)");
        let mut i = Cell::new(0usize);
        w.init_range(
            || {
                let idx = i.get();
                i.set(idx + 1);
                weights_src[idx]
            },
            w.pt_start as usize,
            w.pt_end as usize,
        );
        w.flush();
    }

    pub fn update_bias(&mut self, bias_src: &[f32]) {
        let b = Arc::get_mut(&mut self.bias)
            .expect("update_bias called after sharing bias (Arc not unique)");
        let mut i = Cell::new(0usize);
        b.init_range(
            || {
                let idx = i.get();
                i.set(idx + 1);
                bias_src[idx]
            },
            b.pt_start as usize,
            b.pt_end as usize,
        );
        b.flush();
    }

    pub fn add_to_hash_table(&mut self, length: usize, id: usize) {
        let hashes: Vec<i32> = {
            match HashFunction {
                1 => self.wta_hasher
                    .as_ref().expect("wta_hasher not initialized")
                    .get_hash_psl(&self.weights),

                2 => {
                    let node_ref = &self.nodes[id];
                    self.dwta_hasher
                        .as_ref().expect("dwta_hasher not initialized")
                        .get_hash_easy_pslarray(&self.weights, length, TOPK as usize, node_ref)
                }

                3 => {
                    let binids = self.binids.as_ref().expect("binids not initialized");
                    self.min_hasher
                        .as_ref().expect("min_hasher not initialized")
                        .get_hash_easy_pslarray(binids, &self.weights, length, TOPK as usize)
                }

                4 => self.srp
                    .as_ref().expect("srp not initialized")
                    .get_hash(&self.weights, length),

                _ => unreachable!("Unsupported HashFunction"),
            }
        };

        let hash_indices: Vec<usize> = self.hash_tables.hashes_to_index(&hashes);
        let bucket_indices_i32: Vec<i32> = self.hash_tables.add(&hash_indices, (id as i32) + 1);

        let node = &mut self.nodes[id];
        node.indices_in_tables  = Some(hash_indices);
        node.indices_in_buckets = Some(bucket_indices_i32.iter().map(|&x| x as usize).collect());
    }

    pub fn get_node_by_id(&mut self, node_id: usize) -> &mut Node {
        assert!(node_id < self.no_of_nodes, "nodeID less than _noOfNodes");
        &mut self.nodes[node_id]
    }

    pub fn get_all_nodes(&mut self) -> &mut [Node] {
        &mut self.nodes[..]
    }

    pub fn get_node_count(&self) -> usize {
        self.no_of_nodes
    }

    pub fn get_normalization_constant(&self, input_id: usize) -> f32 {
        assert!(
            matches!(self.node_type, NodeType::Softmax),
            "Error Call to Normalization Constant for non - softmax layer"
        );
        self.normalization_constants.as_ref().unwrap()[input_id]
    }

    fn inner_product(&self, index1: &[usize], value1: &[f32], len1: usize, node_id: usize) -> f32 {
        let mut total = 0.0f32;
        let base = (node_id * self.previous_layer_num_of_nodes) % self.weights.batch;
        for i in 0..len1 {
            total += value1[i] * self.weights.get(base + index1[i]);
        }
        total
    }

    fn _collision(hashes: &[i32], table_hashes: &[i32], k: usize, l: usize) -> f32 {
        let mut cp = 0usize;
        let mut i = 0usize;
        while i < l {
            let mut tmp = 0usize;
            for j in i..(i + k) {
                if hashes[j] == table_hashes[j] {
                    tmp += 1;
                }
            }
            if tmp == k {
                cp += 1;
            }
            i += k;
        }
        (cp as f32) / (l as f32 / k as f32)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn query_active_node_and_compute_activations(
        &mut self,
        active_nodes_per_layer: &mut [Vec<usize>],
        active_values_per_layer: &mut [Vec<f32>],
        lengths: &mut [usize],
        layer_index: usize,
        input_id: usize,
        label: &[usize],
        sparsity: f32,
        _iter: i32,
        full_weights: &[Arc<CacheEntry<f32>>],
        full_bias: &[Arc<CacheEntry<f32>>],
    ) -> usize {
        let mut len: usize;
        let mut in_flag: usize = 0;
        
        if (sparsity - 1.0).abs() < std::f32::EPSILON {
            len = self.no_of_nodes;
            lengths[layer_index + 1] = len;
            active_nodes_per_layer[layer_index + 1] = (0..len).collect();
        } else {
            if Mode == 1 {
                let hashes: Vec<i32> = match HashFunction {
                    1 => self
                        .wta_hasher
                        .as_ref()
                        .expect("wta_hasher not initialized")
                        .get_hash(&active_values_per_layer[layer_index]),
                    2 => {
                            let idx = active_nodes_per_layer[layer_index].clone();
                            self.dwta_hasher
                                .as_ref()
                                .expect("dwta_hasher not initialized")
                                .get_hash(&idx, &active_values_per_layer[layer_index], lengths[layer_index])
                        }
                    3 => {
                        let binids = self.binids.as_ref().expect("binids not initialized");
                        self.min_hasher
                            .as_ref()
                            .expect("min_hasher not initialized")
                            .get_hash_easy(binids, &active_values_per_layer[layer_index], lengths[layer_index], TOPK as usize)
                    }
                    4 => self
                        .srp
                        .as_ref()
                        .expect("srp not initialized")
                        .get_hash_sparse(
                            &active_nodes_per_layer[layer_index],
                            &active_values_per_layer[layer_index],
                            lengths[layer_index],
                        ),
                    _ => unreachable!(),
                };

                let hash_indices = self.hash_tables.hashes_to_index(&hashes);
                let actives = self.hash_tables.retrieve_raw(&hash_indices);

                let mut counts: HashMap<usize, usize> = HashMap::new();

                // Fix: Don't give labels unfair advantage in sparse selection
                // Labels should compete fairly with hash-selected nodes
                if matches!(self.node_type, NodeType::Softmax) && !label.is_empty() {
                    for &lab in label {
                        counts.insert(lab, 1);  // Fair initial count, not self.l
                    }
                }

                for i in 0..self.l {
                    let bucket = &actives[i];
                    if bucket.is_empty() {
                        continue;
                    }
                    let upto = std::cmp::min(BUCKETSIZE as usize, bucket.len());
                    for j in 0..upto {
                        let temp_id = bucket[j] - 1;
                        if temp_id >= 0 {
                            let e = counts.entry(temp_id as usize).or_insert(0);
                            *e += 1;
                        } else {
                            break;
                        }
                    }
                }

                let mut vect: Vec<usize> = Vec::new();
                for (k, v) in counts.iter() {
                    // Fix: Use >= instead of > for proper thresholding
                    if *v >= THRESH as usize {
                        vect.push(*k);
                    }
                }

                len = vect.len();
                lengths[layer_index + 1] = len;
                active_nodes_per_layer[layer_index + 1] = vect;
                

                
                in_flag = len;
            } else if Mode == 4 {
                let hashes: Vec<i32> = match HashFunction {
                    1 => self
                        .wta_hasher
                        .as_ref()
                        .expect("wta_hasher not initialized")
                        .get_hash(&active_values_per_layer[layer_index]),
                    2 => {
                            let idx = active_nodes_per_layer[layer_index].clone();
                            self.dwta_hasher
                                .as_ref()
                                .expect("dwta_hasher not initialized")
                                .get_hash(&idx, &active_values_per_layer[layer_index], lengths[layer_index])
                    }
                    3 => {
                        let binids = self.binids.as_ref().expect("binids not initialized");
                        self.min_hasher
                            .as_ref()
                            .expect("min_hasher not initialized")
                            .get_hash_easy(binids, &active_values_per_layer[layer_index], lengths[layer_index], TOPK as usize)
                    }
                    4 => self
                        .srp
                        .as_ref()
                        .expect("srp not initialized")
                        .get_hash_sparse(
                            &active_nodes_per_layer[layer_index],
                            &active_values_per_layer[layer_index],
                            lengths[layer_index],
                        ),
                    _ => unreachable!(),
                };

                let hash_indices = self.hash_tables.hashes_to_index(&hashes);
                let actives = self.hash_tables.retrieve_raw(&hash_indices);

                let mut counts: HashMap<usize, usize> = HashMap::new();

                // Fix: Don't give labels unfair advantage in sparse selection (Mode 4)
                // Labels should compete fairly with hash-selected nodes  
                if matches!(self.node_type, NodeType::Softmax) && !label.is_empty() {
                    for &lab in label {
                        counts.insert(lab, 1);  // Fair initial count, not self.l
                    }
                }

                for i in 0..self.l {
                    let bucket = &actives[i];
                    if bucket.is_empty() {
                        continue;
                    }
                    let upto = std::cmp::min(BUCKETSIZE as usize, bucket.len());
                    for j in 0..upto {
                        let temp_id = bucket[j] - 1;
                        if temp_id >= 0 {
                            let e = counts.entry(temp_id as usize).or_insert(0);
                            *e += 1;
                        } else {
                            break;
                        }
                    }
                }

                in_flag = counts.len();

                if counts.len() < 1500 {
                    let mut rng = StdRng::seed_from_u64(0);
                    let start = rng.gen_range(0..self.no_of_nodes);
                    for i in start..self.no_of_nodes {
                        if counts.len() >= 1000 {
                            break;
                        }
                        let key = self.rand_node[i];
                        counts.entry(key).or_insert(0);
                    }
                    if counts.len() < 1000 {
                        for i in 0..self.no_of_nodes {
                            if counts.len() >= 1000 {
                                break;
                            }
                            let key = self.rand_node[i];
                            counts.entry(key).or_insert(0);
                        }
                    }
                }

                len = counts.len();
                lengths[layer_index + 1] = len;
                let mut next = Vec::with_capacity(len);
                for (k, _) in counts.into_iter() {
                    next.push(k);
                }
                active_nodes_per_layer[layer_index + 1] = next;
            } else if Mode == 2 && matches!(self.node_type, NodeType::Softmax) {
                len = (self.no_of_nodes as f32 * sparsity).floor() as usize;
                lengths[layer_index + 1] = len;
                let mut bs = vec![false; MAPLEN as usize];
                let mut out = Vec::with_capacity(len);
                let mut tmpsize = 0usize;

                if !label.is_empty() {
                    for &lab in label {
                        out.push(lab);
                        bs[lab] = true;
                    }
                    tmpsize = label.len();
                }

                let mut rng = StdRng::seed_from_u64(0);
                while tmpsize < len {
                    let v = rng.gen_range(0..self.no_of_nodes);
                    if !bs[v] {
                        out.push(v);
                        bs[v] = true;
                        tmpsize += 1;
                    }
                }

                active_nodes_per_layer[layer_index + 1] = out;
            } else if Mode == 3 && matches!(self.node_type, NodeType::Softmax) {
                len = (self.no_of_nodes as f32 * sparsity).floor() as usize;
                lengths[layer_index + 1] = len;

                let mut sortw: Vec<(f32, usize)> = Vec::with_capacity(self.no_of_nodes);
                for s in 0..self.no_of_nodes {
                    let mut tmp = self.inner_product(
                        &active_nodes_per_layer[layer_index],
                        &active_values_per_layer[layer_index],
                        lengths[layer_index],
                        s,
                    );
                    let bias_batch = self.bias.batch;
                    tmp += self.bias.get(s % bias_batch);
                    if label.contains(&s) {
                        sortw.push((-1_000_000_000.0f32, s));
                    } else {
                        sortw.push((-tmp, s));
                    }
                }

                sortw.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                let mut out = Vec::with_capacity(len);
                for i in 0..len {
                    let id = sortw[i].1;
                    out.push(id);
                    if label.contains(&id) {
                        in_flag = 1;
                    }
                }
                active_nodes_per_layer[layer_index + 1] = out;
            } else {
                len = 0;
                lengths[layer_index + 1] = 0;
                active_nodes_per_layer[layer_index + 1].clear();
            }
        }

        active_values_per_layer[layer_index + 1] = vec![0.0f32; lengths[layer_index + 1]];
        let next_len = lengths[layer_index + 1];

        let mut max_value = 0.0f32;
        if matches!(self.node_type, NodeType::Softmax) {
            if let Some(ref mut norms) = self.normalization_constants {
                norms[input_id] = 0.0;
            }
        }

        for i in 0..next_len {
            let node_id = active_nodes_per_layer[layer_index + 1][i];
            
            // Ensure node_id is within bounds for this layer
            let node_id = node_id.min(self.nodes.len().saturating_sub(1));
            
            let worker_id = (node_id * self.previous_layer_num_of_nodes) / self.weights.batch;

            // Ensure worker_id is within bounds for distributed storage
            let worker_id = worker_id.min(full_weights.len().saturating_sub(1));
            debug_assert!(worker_id < full_weights.len());
            debug_assert!(worker_id < full_bias.len());

            let w_guard = full_weights[worker_id].data.lock().unwrap();
            let b_guard = full_bias[worker_id].data.lock().unwrap();
            let lw: &[f32] = &w_guard[..];
            let lb: &[f32] = &b_guard[..];

            let val = self.nodes[node_id].get_activation(
                &active_nodes_per_layer[layer_index],
                &active_values_per_layer[layer_index],
                lengths[layer_index],
                input_id,
                lw,
                lb,
            );
            active_values_per_layer[layer_index + 1][i] = val;

            if matches!(self.node_type, NodeType::Softmax) && val > max_value {
                max_value = val;
            }
        }

        if matches!(self.node_type, NodeType::Softmax) {
            // Match C++ implementation exactly - store unnormalized exp values
            // Normalization happens later in compute_extra_stats_for_softmax
            if let Some(ref mut norms) = self.normalization_constants {
                norms[input_id] = 0.0; // Reset normalization constant
            }
            
            for i in 0..next_len {
                let real_activation = (active_values_per_layer[layer_index + 1][i] - max_value).exp();
                active_values_per_layer[layer_index + 1][i] = real_activation;
                
                if let Some(ref mut norms) = self.normalization_constants {
                    norms[input_id] += real_activation;
                }
                
                let node_id = active_nodes_per_layer[layer_index + 1][i];
                let node_id = node_id.min(self.nodes.len().saturating_sub(1));
                
                self.nodes[node_id]
                    .set_last_activation(input_id, real_activation);
            }
        }

        in_flag
    }

    pub fn save_weights(&self, _file: &str) {
    }
}
