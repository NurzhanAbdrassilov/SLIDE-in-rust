use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand::seq::SliceRandom;

use crate::psl_array::PSLArray;

#[derive(Debug)]
pub struct DensifiedWtaHash {
    rand_hash: [u32; 2],
    randa: u32,
    num_hashes: usize,
    range_pow: usize,
    log_num_hash: u32,
    indices: Vec<usize>,
    pos: Vec<i32>,
    permute: usize,
}

impl DensifiedWtaHash {
    pub fn new(num_hashes: usize, no_of_bits_to_hash: usize) -> Self {
        assert!(num_hashes > 0, "num_hashes must be > 0");
        assert!(no_of_bits_to_hash > 0, "no_of_bits_to_hash must be > 0");

        let mut rng = StdRng::seed_from_u64(33_111);

        let binsize: usize = crate::config::binsize as usize;

        let permute = ((num_hashes * binsize) as f64 / no_of_bits_to_hash as f64).ceil() as usize;

        let mut base: Vec<usize> = (0..no_of_bits_to_hash).collect();
        let mut indices = vec![0usize; no_of_bits_to_hash * permute];
        let mut pos     = vec![0i32;  no_of_bits_to_hash * permute];

        for p in 0..permute {
            base.shuffle(&mut rng);
            for j in 0..no_of_bits_to_hash {
                let slot      = p * no_of_bits_to_hash + base[j];
                let linear    = p * no_of_bits_to_hash + j;
                indices[slot] = linear / binsize;
                pos[slot]     = (linear % binsize) as i32;
            }
        }

        let log_num_hash = ((num_hashes as f64).log2().ceil() as u32).max(1);
        let randa: u32 = rng.gen::<u32>() | 1;
        let rand_hash  = [rng.gen::<u32>() | 1, rng.gen::<u32>() | 1];

        Self {
            rand_hash,
            randa,
            num_hashes,
            range_pow: no_of_bits_to_hash,
            log_num_hash: log_num_hash,
            indices,
            pos,
            permute,
        }
    }

    pub fn get_hash_easy_pslarray(
        &self,
        data: &PSLArray<f32>,
        data_len: usize,
        _topk: usize,
        node: &crate::node::Node,
    ) -> Vec<i32> {
        let h = self.num_hashes;
        let mut hashes = vec![i32::MIN; h];
        let mut values = vec![f32::MIN; h];
        let mut out    = vec![0i32;     h];

        let idx      = node.id_in_layer * node.dim;
        let offset   = idx % data.batch;
        let entry    = data.get_raw_arr(idx, true, false);
        let guard    = entry.data.lock().unwrap();
        let data_row = &guard[offset..];

        let len = data_len.min(self.range_pow);

        for p in 0..self.permute {
            let bin_base = p * self.range_pow;
            for i in 0..len {
                let inner      = bin_base + i;
                let binid      = self.indices[inner];
                let loc        = data_row[i];
                if binid < h && values[binid] < loc {
                    values[binid] = loc;
                    hashes[binid] = self.pos[inner];
                }
            }
        }

        self.densify_into(&mut hashes, &mut out);
        out
    }

    pub fn get_hash_easy_float(&self, data: &[f32], data_len: usize, _topk: usize) -> Vec<i32> {
        let h = self.num_hashes;
        let mut hashes = vec![i32::MIN; h];
        let mut values = vec![f32::MIN; h];
        let mut out    = vec![0i32;     h];

        let len = data_len.min(self.range_pow);

        for p in 0..self.permute {
            let bin_base = p * self.range_pow;
            for i in 0..len {
                let inner = bin_base + i;
                let binid = self.indices[inner];
                let loc   = data[i];
                if binid < h && values[binid] < loc {
                    values[binid] = loc;
                    hashes[binid] = self.pos[inner];
                }
            }
        }

        self.densify_into(&mut hashes, &mut out);
        out
    }

    pub fn get_hash(&self, indices: &[usize], data: &[f32], data_len: usize) -> Vec<i32> {
        let h = self.num_hashes;
        let mut hashes = vec![i32::MIN; h];
        let mut values = vec![f32::MIN; h];
        let mut out    = vec![0i32;     h];

        let take = data_len.min(indices.len()).min(data.len());

        for p in 0..self.permute {
            let bin_base = p * self.range_pow;

            for i in 0..take {
                let feat = indices[i];
                if feat >= self.range_pow {
                    continue;
                }
                let inner = bin_base + feat;
                let binid = self.indices[inner];
                if binid < h {
                    let val = data[i];
                    if values[binid] < val {
                        values[binid] = val;
                        hashes[binid] = self.pos[inner];
                    }
                }
            }
        }

        self.densify_into(&mut hashes, &mut out);
        out
    }

    #[inline]
    fn densify_into(&self, hashes: &mut [i32], out: &mut [i32]) {
        for i in 0..self.num_hashes {
            let mut next = hashes[i];
            if next != i32::MIN {
                out[i] = next;
                continue;
            }
            let mut cnt = 0usize;
            while next == i32::MIN && cnt < 100 {
                cnt += 1;
                let idx = self
                    .get_rand_double_hash(i, cnt)
                    .min(self.num_hashes - 1);
                next = hashes[idx];
            }
            out[i] = next;
        }
    }

    #[inline]
    fn get_rand_double_hash(&self, binid: usize, count: usize) -> usize {
        let tohash = (((binid + 1) as u32) << 6).wrapping_add(count as u32);
        (((self.rand_hash[0].wrapping_mul(tohash)) << 3) >> (32 - self.log_num_hash)) as usize
    }
}
