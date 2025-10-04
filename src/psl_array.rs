use std::collections::HashMap;
use std::mem;
use std::slice;
use std::sync::{Arc, Mutex};
use rand::Rng;

use crate::types::{kv_key_t, kv_val_datatype_t, kv_val_t};
use crate::scl::{read_key, write_kv, commit_tx};
use crate::scl::block_storage_status;

pub const DEFAULT_BATCH_SIZE: usize = 1 << 25;
pub const DEFAULT_CACHE_SIZE: usize = 1 << 5;
pub const MAX_BATCH_SIZE: usize = 1 << 5;

#[derive(Clone, Copy, Debug)]
pub struct Range {
    pub start: i32,
    pub end: i32,
}

pub struct CacheEntry<T: Copy> {
    pub data: Mutex<Vec<T>>,
    pub key: kv_key_t,
    pub batch: usize,
    pub write_on_drop: bool,
}

impl<T: Copy> Drop for CacheEntry<T> {
    fn drop(&mut self) {
        if self.write_on_drop {
            let ctx = crate::scl::new_context(0);
            let guard = self.data.lock().unwrap();
            let byte_slice = unsafe {
                slice::from_raw_parts(
                    guard.as_ptr() as *const u8,
                    guard.len() * mem::size_of::<T>(),
                )
            };
            let val = kv_val_t {
                data: byte_slice.to_vec(),
                dtype: kv_val_datatype_t::BYTES,
                ts: 0,
            };
            write_kv(ctx, &self.key, &val);
            assert_eq!(commit_tx(ctx), block_storage_status::Accept);
        }
    }
}

pub struct PSLCache<T: Default + Copy> {
    cache: Mutex<HashMap<usize, Arc<CacheEntry<T>>>>,
    max_size: usize,
    batch: usize,
}

impl<T: Default + Copy> PSLCache<T> {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CACHE_SIZE, DEFAULT_BATCH_SIZE)
    }

    pub fn with_capacity(max_size: usize, batch: usize) -> Self {
        PSLCache {
            cache: Mutex::new(HashMap::new()),
            max_size,
            batch,
        }
    }

    pub fn clear(&self) {
        self.cache.lock().unwrap().clear();
    }

    pub fn init(&mut self, max_size: usize, batch: usize) {
        self.max_size = max_size;
        self.batch = batch;
    }

    pub fn put(&self, _key: &kv_key_t, entry: Arc<CacheEntry<T>>, index: usize) {
        let idx = self.batch * (index / self.batch);
        let mut c = self.cache.lock().unwrap();
        while c.len() >= self.max_size {
            if let Some((&first, _)) = c.iter().next() {
                c.remove(&first);
            }
        }
        c.insert(idx, entry);
    }

    pub fn get(
        &self,
        key: kv_key_t,
        index: usize,
        read_only: bool,
        new_read: bool,
    ) -> Arc<CacheEntry<T>> {
        let idx = self.batch * (index / self.batch);
        if let Some(entry) = self.cache.lock().unwrap().get(&idx) {
            return entry.clone();
        }

        let ctx = crate::scl::new_context(0);

        let v: kv_val_t = if new_read {
            kv_val_t { data: Vec::new(), dtype: kv_val_datatype_t::NOT_FOUND, ts: 0 }
        } else {
            let boxed = read_key(ctx, &key);
            *boxed
        };

        let mut data = vec![T::default(); self.batch];
        if v.dtype != kv_val_datatype_t::NOT_FOUND {
            let data_ptr = v.data.as_ptr() as *const T;
            unsafe {
                std::ptr::copy_nonoverlapping(data_ptr, data.as_mut_ptr(), self.batch);
            }
        }

        let write_on_drop = !read_only;
        let entry = Arc::new(CacheEntry {
            data: Mutex::new(data),
            key: key.clone(),
            batch: self.batch,
            write_on_drop,
        });
        self.put(&key, entry.clone(), index);
        entry
    }
}

pub struct PSLArray<T: Default + Copy> {
    pub size: usize,
    pub batch: usize,
    pub name: String,

    offset: usize,
    cache: PSLCache<T>,

    pub pt_start: i32,
    pub pt_end: i32,
    pub num_workers: usize,
    pub worker_id: usize,
    pub partition_map: HashMap<usize, Range>,
}

impl<T: Default + Copy> PSLArray<T> {
    pub fn default() -> Self {
        PSLArray::new("hi".to_string(), 4096, DEFAULT_BATCH_SIZE)
    }

    pub fn new(name: String, size: usize, batch: usize) -> Self {
        let mut cache = PSLCache::with_capacity(DEFAULT_CACHE_SIZE, batch);
        let new_batch = std::cmp::min(size, batch);
        cache.init(size, new_batch);
        PSLArray {
            name,
            size,
            offset: 0,
            batch: new_batch,
            cache,
            pt_start: 0,
            pt_end: size as i32,
            num_workers: 0,
            worker_id: 0,
            partition_map: HashMap::new(),
        }
    }

    pub fn with_workers(
        name: String,
        size: usize,
        num_workers: usize,
        worker_id: usize,
        div: usize,
    ) -> Self {
        let mut partition_map = HashMap::new();

        let x = (size + div - 1) / div;
        let y = (x + num_workers - 1) / num_workers;
        let base = y * div;
        let remainder = size - base * (num_workers - 1);

        let mut start = 0usize;
        for w in 0..num_workers {
            let worker_size = if w == num_workers - 1 { remainder } else { base };
            let end = start + worker_size;
            partition_map.insert(w, Range { start: start as i32, end: end as i32 });
            start = end;
        }

        let range = partition_map[&worker_id];
        let batch = base;

        let mut cache = PSLCache::with_capacity(DEFAULT_CACHE_SIZE, batch);
        cache.init(DEFAULT_CACHE_SIZE, batch);

        PSLArray {
            name,
            size,
            offset: 0,
            batch,
            cache,
            pt_start: range.start,
            pt_end: range.end,
            num_workers,
            worker_id,
            partition_map,
        }
    }

    pub fn with_init<F>(name: String, size: usize, func: F, batch: usize) -> Self
    where
        F: Fn() -> T,
    {
        let mut arr = PSLArray::new(name, size, batch);
        arr.init(func);
        arr
    }

    pub fn init_psl_array(&mut self, size: usize, name: String, offset: usize, batch: usize) {
        self.size = size;
        self.name = name;
        self.offset = offset;
        self.batch = batch;
        self.cache.init(DEFAULT_CACHE_SIZE, self.batch);
        println!("init_psl_array: size {}, this.size {}", size, self.size);
    }

    pub fn get_size(&self) -> usize { self.size }

    pub fn offset_array(&self, off: usize) -> Self {
        self.flush();
        let mut ret = PSLArray::new(self.name.clone(), self.size, self.batch);
        ret.set_offset(self.offset + off);
        ret.cache.init(DEFAULT_CACHE_SIZE, self.batch);
        ret
    }

    fn get_effective_key(&self, mut index: usize) -> kv_key_t {
        index += self.offset;
        assert!(index < self.size, "{}: index {} >= size {}", self.name, index, self.size);

        index = self.batch * (index / self.batch);

        if index < self.batch * (self.size / self.batch) {
            format!("{}_{}_{}", self.name, index, index + self.batch)
        } else {
            format!("{}_{}_{}", self.name, self.batch * (self.size / self.batch), self.size)
        }
    }

    pub fn get(&self, idx: usize) -> T {
        assert!(idx < self.size);
        let key = self.get_effective_key(idx);
        let entry = self.cache.get(key, idx, true, false);
        let guard = entry.data.lock().unwrap();
        guard[idx % self.batch]
    }

    pub fn get_raw_arr(&self, idx: usize, read_only: bool, new_read: bool) -> Arc<CacheEntry<T>> {
        assert!(idx < self.size);
        let key = self.get_effective_key(idx);
        self.cache.get(key, idx, read_only, new_read)
    }

   pub fn get_arr(&self, idx: usize) -> Arc<CacheEntry<T>> {
        self.get_raw_arr(idx, false, false)
    }
    pub fn get_arr_batch_idx_raw(&self, w: usize, read_only: bool, new_read: bool) -> Arc<CacheEntry<T>> {
        let start = self.partition_map[&w].start as usize;
        self.get_raw_arr(start, read_only, new_read)
    }

    pub fn get_arr_batch_idx(&self, w: usize) -> Arc<CacheEntry<T>> {
        self.get_arr_batch_idx_raw(w, false, false)
    }

    pub fn flush_batch_idx(&self, entry: Arc<CacheEntry<T>>, w: usize) {
        let range = &self.partition_map[&w];
        let key = self.get_effective_key(range.start as usize);

        let guard = entry.data.lock().unwrap();
        let byte_slice = unsafe {
            slice::from_raw_parts(
                guard.as_ptr() as *const u8,
                self.batch * mem::size_of::<T>(),
            )
        };
        let val = kv_val_t {
            data: byte_slice.to_vec(),
            dtype: kv_val_datatype_t::BYTES,
            ts: 0,
        };
        let ctx = crate::scl::new_context(0);
        write_kv(ctx, &key, &val);
        assert_eq!(commit_tx(ctx), block_storage_status::Accept);
    }

    pub fn set(&self, v: T, idx: usize) {
        assert!(idx < self.size);
        let key = self.get_effective_key(idx);
        let entry = self.cache.get(key, idx, false, false);
        let mut guard = entry.data.lock().unwrap();
        guard[idx % self.batch] = v;
    }

    pub fn init<F>(&mut self, func: F)
    where
        F: Fn() -> T,
    {
        assert!(self.size != 0);
        let mut i = self.offset;
        while i < self.size {
            let len = std::cmp::min(self.batch, self.size - i);
            let entry = self.cache.get(self.get_effective_key(i), i, false, true);
            let mut guard = entry.data.lock().unwrap();
            for j in 0..len {
                guard[j] = func();
            }
            i += self.batch;
        }
    }

    pub fn init_range<F>(&mut self, func: F, start: usize, end: usize)
    where
        F: Fn() -> T,
    {
        assert!(end <= self.size);
        let mut i = start;
        while i < end {
            let len = std::cmp::min(self.batch, end - i);
            let entry = self.cache.get(self.get_effective_key(i), i, false, true);
            let mut guard = entry.data.lock().unwrap();
            for j in 0..len {
                guard[j] = func();
            }
            i += self.batch;
        }
    }

    pub fn flush(&self) {
        self.cache.clear();
    }

    pub fn set_offset(&mut self, off: usize) {
        self.offset = off;
    }

    pub fn get_worker_range(&self, worker_idx: usize) -> Option<Range> {
        self.partition_map.get(&worker_idx).cloned()
    }

    pub fn shuffle<R: Rng>(&mut self, rng: &mut R) {
        for i in (1..self.size).rev() {
            let j = rng.gen_range(0..=i);
            let tmp = self.get(i);
            self.set(self.get(j), i);
            self.set(tmp, j);
        }
    }

    pub fn begin(&self) -> PSLArrayIterator<T> {
        PSLArrayIterator::new(self.name.clone(), self.offset, self.size, self.batch, &self.cache, self.offset)
    }

    pub fn end(&self) -> PSLArrayIterator<T> {
        PSLArrayIterator::new(self.name.clone(), self.size, self.size, self.batch, &self.cache, self.offset)
    }
}


pub struct PSLArrayIterator<'a, T: Default + Copy> {
    name: String,
    curr_idx: usize,
    end_idx: usize,
    size: usize,
    batch: usize,
    offset: usize,
    cache: &'a PSLCache<T>,
}

impl<'a, T: Default + Copy> PSLArrayIterator<'a, T> {
    pub fn new(
        name: String,
        curr_idx: usize,
        end_idx: usize,
        batch: usize,
        cache: &'a PSLCache<T>,
        offset: usize,
    ) -> Self {
        PSLArrayIterator {
            name,
            curr_idx,
            end_idx,
            size: end_idx,
            batch,
            offset,
            cache,
        }
    }

    fn get_effective_key(&self, mut index: usize) -> kv_key_t {
        assert!(index < self.size);
        index = self.batch * (index / self.batch);
        if index < self.batch * (self.size / self.batch) {
            format!("{}_{}_{}", self.name, index + self.offset, index + self.offset + self.batch)
        } else {
            format!("{}_{}_{}", self.name, self.batch * (self.size / self.batch), self.size)
        }
    }
}

impl<'a, T: Default + Copy> Iterator for PSLArrayIterator<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr_idx >= self.end_idx {
            return None;
        }
        let key = self.get_effective_key(self.curr_idx);
        let entry = self.cache.get(key, self.curr_idx, true, false);
        let guard = entry.data.lock().unwrap();
        let val = guard[self.curr_idx % self.batch];
        self.curr_idx += 1;
        Some(val)
    }
}
