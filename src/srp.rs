use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use crate::psl_array::PSLArray;
use rand::seq::SliceRandom;

pub struct SparseRandomProjection {
    dim: usize,
    num_hashes: usize,
    sam_size: usize,
    rand_bits: Vec<Vec<i16>>,
    indices: Vec<Vec<usize>>,
}

impl SparseRandomProjection {
    pub fn new(dimension: usize, num_hashes: usize, ratio: usize) -> Self {
        let dim = dimension;
        let num_hashes = num_hashes;
        let sam_size = (dim as f64 / ratio as f64).ceil() as usize;

        let mut a: Vec<usize> = (0..dim).collect();

        let mut shuf = StdRng::seed_from_u64(33111);
        let mut signs = StdRng::seed_from_u64(0);

        let mut rand_bits: Vec<Vec<i16>> = Vec::with_capacity(num_hashes);
        let mut indices: Vec<Vec<usize>> = Vec::with_capacity(num_hashes);

        for _ in 0..num_hashes {
            a.shuffle(&mut shuf);
            let mut curr_rand_bits: Vec<i16> = Vec::with_capacity(sam_size);
            let mut curr_indices: Vec<usize> = Vec::with_capacity(sam_size);

            for j in 0..sam_size {
                curr_indices.push(a[j]);
                let curr: u32 = signs.gen();
                if curr % 2 == 0 {
                    curr_rand_bits.push(1);
                } else {
                    curr_rand_bits.push(-1);
                }
            }
            curr_indices.sort();
            rand_bits.push(curr_rand_bits);
            indices.push(curr_indices);
        }

        SparseRandomProjection {
            dim,
            num_hashes,
            sam_size,
            rand_bits,
            indices,
        }
    }

    pub fn get_hash(&self, vector: &PSLArray<f32>, length: usize) -> Vec<i32> {
        // length should be = to dim
        let mut hashes: Vec<i32> = Vec::with_capacity(self.num_hashes);

        for i in 0..self.num_hashes {
            let mut s: f64 = 0.0;
            for j in 0..self.sam_size {
                let v: f32 = vector.get(self.indices[i][j]);
                if self.rand_bits[i][j] >= 0 {
                    s += v as f64;
                } else {
                    s -= v as f64;
                }
            }
            hashes.push(if s >= 0.0 { 0 } else { 1 });
        }
        hashes
    }

    pub fn get_hash_array(&self, vector: &[f32], length: usize) -> Vec<i32> {
        // length should be = to dim
        let mut hashes = Vec::with_capacity(self.num_hashes);
        for i in 0..self.num_hashes {
            let mut s: f64 = 0.0;
            for j in 0..self.sam_size {
                let v = vector[self.indices[i][j]];
                if self.rand_bits[i][j] >= 0 {
                    s += v as f64;
                } else {
                    s -= v as f64;
                }
            }
            hashes.push(if s >= 0.0 { 0 } else { 1 });
        }
        hashes
    }
    pub fn get_hash_sparse(&self, indices: &[usize], values: &[f32], length: usize) -> Vec<i32> {
        let mut hashes: Vec<i32> = Vec::with_capacity(self.num_hashes);

        for p in 0..self.num_hashes {
            let mut s: f64 = 0.0;
            let mut i: usize = 0;
            let mut j: usize = 0;
            while i < length && j < self.sam_size {
                if indices[i] == self.indices[p][j] {
                    let v: f32 = values[i];
                    if self.rand_bits[p][j] >= 0 {
                        s += v as f64;
                    } else {
                        s -= v as f64;
                    }
                    i += 1;
                    j += 1;
                } else if indices[i] < self.indices[p][j] {
                    i += 1;
                } else {
                    j += 1;
                }
            }
            hashes.push(if s >= 0.0 { 0 } else { 1 });
        }
        hashes
    }
}