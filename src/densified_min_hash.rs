use crate::murmurhash::murmur_hash;
use crate::psl_array::PSLArray;
use std::cell::RefCell;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::rc::Rc;

const INT_MIN: i32 = i32::MIN;

pub struct DensifiedMinhash {
    rand_hash: [u32; 2],
    randa: u32,
    num_hashes: usize,
    range_pow: usize,
    log_num_hash: u32,
}

impl DensifiedMinhash {
    pub fn new(num_hashes: usize, no_of_bits_to_hash: usize) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(33111);
        let mut next_odd = || {
            let mut val: u32 = rng.gen();
            if val % 2 == 0 {
                val += 1;
            }
            val
        };

        DensifiedMinhash {
            rand_hash: [next_odd(), next_odd()],
            randa: next_odd(),
            num_hashes: num_hashes,
            range_pow: no_of_bits_to_hash,
            log_num_hash: (num_hashes as f64).log2().ceil() as u32,
        }
    }

    pub fn get_map(&self, n: usize, binids: &mut [usize]) {
        let range = 1 << self.range_pow;
        // binsize is the number of times the range is larger than the total number of hashes we need.
        let binsize = (range as f64 / self.num_hashes as f64).ceil() as usize;

        for i in 0..n {
            let mut h = i as u32;
            h = h.wrapping_mul(self.randa);
            h ^= h >> 13;
            h = h.wrapping_mul(0x85ebca6b);

            let key = i.to_ne_bytes();
//        unsigned int curhash = (unsigned int)(((unsigned int)h*i) << 5);
            let mut curhash = murmur_hash(&key, key.len() as u32, self.randa);
            curhash &= (1 << self.range_pow) - 1;
            binids[i] = (curhash as usize) / binsize;
        }
    }

    pub fn get_hash_easy_pslarray(
        &self,
        binids: &[usize],
        data: &PSLArray<f32>,
        data_len: usize,
        top_k: usize,
    ) -> Vec<i32> {
           // binsize is the number of times the range is larger than the total number of hashes we need.
// read the data and add it to priority queue O(dlogk approx 7d) with index as key and values as priority value, get topk index O(1) and apply minhash on retuned index.

        let mut pq = BinaryHeap::with_capacity(top_k);
        for i in 0..top_k.min(data_len) {
            pq.push(HeapItem(i, data.get(i)));
        }
        for i in top_k.min(data_len)..data_len {
            pq.push(HeapItem(i, data.get(i)));
            pq.pop();
        }

        let mut hashes = vec![INT_MIN; self.num_hashes];
        for HeapItem(index, _) in pq.into_sorted_vec() {
            let binid = binids[index];
            if hashes[binid] < index as i32 {
                hashes[binid] = index as i32;
            }
        }

        let mut hash_array = vec![0; self.num_hashes];
        for i in 0..self.num_hashes {
            let mut next = hashes[i];
            if next != INT_MIN {
                hash_array[i] = next;
                continue;
            }
            let mut count = 0;
            while next == INT_MIN && count <= 100 {
                count += 1;
                let index = self.get_rand_double_hash(i, count).min(self.num_hashes - 1); // -1 to not go out of boundsd
                next = hashes[index];
            }
            hash_array[i] = next;
        }

        hash_array
    }

    pub fn get_hash_easy(
        &self,
        binids: &[usize],
        data: &[f32],
        data_len: usize,
        top_k: usize,
    ) -> Vec<i32> {
        let mut pq = BinaryHeap::with_capacity(top_k);
        for i in 0..top_k.min(data_len) {
            pq.push(HeapItem(i, data[i]));
        }
        for i in top_k..data_len {
            pq.push(HeapItem(i, data[i]));
            pq.pop();
        }

        let mut hashes = vec![INT_MIN; self.num_hashes];
        for HeapItem(index, _) in pq.into_sorted_vec() {
            let binid = binids[index];
            if hashes[binid] < index as i32 {
                hashes[binid] = index as i32;
            }
        }

        let mut hash_array = vec![0; self.num_hashes];
        for i in 0..self.num_hashes {
            let mut next = hashes[i];
            if next != INT_MIN {
                hash_array[i] = next;
                continue;
            }
            let mut count = 0;
            while next == INT_MIN && count <= 100 {
                count += 1;
                let index = self.get_rand_double_hash(i, count).min(self.num_hashes - 1);
                next = hashes[index];
            }
            hash_array[i] = next;
        }

        hash_array
    }

    pub fn get_hash(
        &self,
        indices: &[usize],
        data: &[f32],
        binids: &[usize],
        data_len: usize,
    ) -> Vec<i32> {
        let mut hashes = vec![INT_MIN; self.num_hashes];

        for &i in indices.iter().take(data_len) {
            let binid = binids[i];
            if hashes[binid] < i as i32 {
                hashes[binid] = i as i32;
            }
        }

        let mut hash_array = vec![0; self.num_hashes];
        for i in 0..self.num_hashes {
            let mut next = hashes[i];
            if next != INT_MIN {
                hash_array[i] = hashes[i];
                continue;
            }
            let mut count = 0;
            while next == INT_MIN && count <= 100 {
                count += 1;
                let index = self.get_rand_double_hash(i, count).min(self.num_hashes - 1);
                next = hashes[index];
            }
            hash_array[i] = next;
        }

        hash_array
    }

    fn get_rand_double_hash(&self, binid: usize, count: usize) -> usize {
        let to_hash = ((binid + 1) << 6) + count;
        let hashed = (self.rand_hash[0].wrapping_mul(to_hash as u32) << 3)
            >> (32 - self.log_num_hash);
        hashed as usize
    }
}

#[derive(Debug)]
struct HeapItem(usize, f32);

impl Eq for HeapItem {}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other.1.partial_cmp(&self.1).unwrap_or(Ordering::Equal)
    }
}
