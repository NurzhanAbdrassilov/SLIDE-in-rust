use rand::{distributions::Uniform, rngs::StdRng, Rng, SeedableRng};
use rand::distributions::Distribution;
use std::f64;
use std::i32;

use crate::bucket::Bucket;
use crate::config::{HashFunction, binsize};

pub struct LSH {
    buckets: Vec<Vec<Bucket>>,
    k: usize,
    l: usize,
    range_pow: usize,
    rand1: Vec<u32>,
}

impl LSH {
    pub fn new(k: usize, l: usize, range_pow: usize) -> LSH {
        let mut buckets = Vec::with_capacity(l);
        for _ in 0..l {
            buckets.push(vec![Bucket::new(); 1 << range_pow]);
        }

        let mut rand1 = Vec::with_capacity(k * l);
        let seed = 33111u64;
        let mut rng = StdRng::seed_from_u64(seed);
        let uniform = Uniform::new_inclusive(1, i32::MAX as u32);
        for _ in 0..(k * l) {
            let mut v = uniform.sample(&mut rng);
            if v % 2 == 0 {
                v += 1;
            }
            rand1.push(v);
        }

        LSH {
            buckets,
            k,
            l,
            range_pow,
            rand1,
        }
    }

    pub fn clear(&mut self) {
        for table in &mut self.buckets {
            *table = vec![Bucket::new(); 1 << self.range_pow];
        }
    }

    pub fn count(&self) {
        for (_j, table) in self.buckets.iter().enumerate() {
            let mut _total = 0;
            for b in table {
                let sz = b.getSize();
                _total += sz;
            }
        }
    }

    pub fn hashes_to_index(&self, hashes: &[i32]) -> Vec<usize> {
        let mut indices = vec![0usize; self.l];
        let bin_log = (binsize as f64).ln().floor() as usize;

        for i in 0..self.l {
            let mut index: u32 = 0;
            for j in 0..self.k {
                let h_i = i * self.k + j;
                let h = hashes[h_i] as u32;
                let slot = i * self.k + j;

                match HashFunction {
                    4 => {
                        index = index.wrapping_add(h << (self.k - 1 - j));
                    }
                    1 | 2 => {
                        index = index.wrapping_add(
                            h << ((self.k - 1 - j) * bin_log)
                        );
                    }
                    _ => {
                        let r = self.rand1[slot];
                        let mut x = r.wrapping_mul(r);
                        x ^= x >> 13;
                        x ^= r;
                        index = index.wrapping_add(x.wrapping_mul(h));
                    }
                }
            }
            // Apply masking for all hash functions to ensure index stays within bucket bounds
            let mask = (1u32 << self.range_pow) - 1;
            index &= mask;
            indices[i] = index as usize;
        }

        indices
    }


    pub fn add(&mut self, indices: &[usize], id: i32) -> Vec<i32> {
        let mut second = Vec::with_capacity(self.l);
        for i in 0..self.l {
            let idx = indices[i];
            let val = self.buckets[i][idx].add(id);
            second.push(val);
        }
        second
    }

    pub fn add_single(&mut self, table: usize, indices: usize, id: i32) -> i32 {
        self.buckets[table][indices].add(id)
    }

   pub fn retrieve_raw(&mut self, indices: &[usize]) -> Vec<Vec<i32>> {
        let mut results = Vec::with_capacity(self.l);
        for i in 0..self.l {
            let idx = indices[i];
            let slice = self.buckets[i][idx].getAll().unwrap_or(&[]);
            results.push(slice.to_vec());
        }
        results
    }


    pub fn retrieve(&self, table: usize, indices: usize, bucket: usize) -> i32 {
        self.buckets[table][indices].retrieve(bucket)
    }
}

