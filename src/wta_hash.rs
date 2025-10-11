use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use std::f64;
use std::i32;
use crate::config::binsize;
use crate::psl_array::PSLArray;

pub struct WtaHash {
    indices: Vec<usize>,
    num_hashes: usize,
    range_pow: usize,
}

impl WtaHash {
    pub fn new(num_hashes: usize, no_of_bits_to_hash: usize) -> Self {

        let range_pow = no_of_bits_to_hash;

        let seed = 33111u64;
        let mut gen = StdRng::seed_from_u64(seed);

        let permute = ((num_hashes as f64 * binsize as f64) / (no_of_bits_to_hash as f64)).ceil() as usize;


        let mut n_array: Vec<usize> = (0..range_pow).collect();
        let mut indices = Vec::with_capacity(range_pow * permute);

        for _ in 0..permute {
            let mut copy = n_array.clone();
            copy.shuffle(&mut gen);
            indices.extend(&copy);
        }

        WtaHash { indices, num_hashes, range_pow }
    }

    pub fn get_hash(&self, data: &[f32]) -> Vec<i32> {
        let bs = binsize as usize;
        let mut hashes = vec![i32::MIN; self.num_hashes];
        let mut values = vec![f32::MIN; self.num_hashes];

        for i in 0..self.num_hashes {
            for j in 0..bs {
                let idx = self.indices[i * bs + j];
                if values[i] < data[idx] {
                    values[i] = data[i * bs + j];
                    hashes[i] = idx as i32;
                }
            }
        }

        hashes
    }

    pub fn get_hash_psl(&self, data: &PSLArray<f32>) -> Vec<i32> {
        let bs = binsize as usize;
        let mut hashes = vec![i32::MIN; self.num_hashes];
        let mut values = vec![f32::MIN; self.num_hashes];

        for i in 0..self.num_hashes {
            for j in 0..bs {
                let idx = self.indices[i * bs + j];
                let d = data.get(idx);
                if values[i] < d {
                    values[i] = data.get(i * bs + j);
                    hashes[i] = idx as i32;
                }
            }
        }

        hashes
    }
}
