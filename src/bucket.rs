use crate::config::{BUCKETSIZE, FIFO};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

#[derive(Clone)]
pub struct Bucket {
    arr: Vec<i32>,
    is_init: i32,
    index: usize,
    counts: usize,
}

impl Bucket {
    pub fn new() -> Self {
        Bucket {
            arr: vec![0; BUCKETSIZE],
            is_init: -1,
            index: 0,
            counts: 0,
        }
    }

    pub fn getTotalCounts(&self) -> usize {
        self.counts
    }

    pub fn getSize(&self) -> usize {
        self.counts
    }

    pub fn add(&mut self, id: i32) -> i32 {
        let mut rng = StdRng::seed_from_u64(0);

        if FIFO {
            self.is_init += 1;
            let idx = self.counts & (BUCKETSIZE - 1);
            self.arr[idx] = id;
            self.counts += 1;
            idx as i32
        } else {
            self.counts += 1;
            if self.index == BUCKETSIZE {
                let randnum = (rng.gen::<u32>() as usize) % self.counts + 1;
                if randnum == 2 {
                    let randind = (rng.gen::<u32>() as usize) % BUCKETSIZE;
                    self.arr[randind] = id;
                    randind as i32
                } else {
                    -1
                }
            } else {
                self.arr[self.index] = id;
                let ret = self.index as i32;
                self.index += 1;
                ret
            }
        }
    }

    pub fn retrieve(&self, indice: usize) -> i32 {
        if indice >= BUCKETSIZE { -1 } else { self.arr[indice] }
    }

    pub fn getAll(&mut self) -> Option<&[i32]> {
        if self.is_init == -1 {
            return None;
        }
        if self.counts < BUCKETSIZE {
            self.arr[self.counts] = -1;
        }
        Some(&self.arr[..])
    }
}
