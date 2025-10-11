use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use crate::types::{kv_key_t, kv_val_datatype_t, kv_val_t};
use rocksdb::{DB, Options, WriteOptions};
use serde::{Serialize, Deserialize};

pub const MAX_VAL_SIZE: usize = std::mem::size_of::<f32>() * (1 << 20);

// RocksDB-based KVS implementation
struct RocksKVS {
    db: DB,
    db_path: String,
    cache: HashMap<String, kv_val_t>,  // In-memory cache for performance
}

impl RocksKVS {
    fn new() -> Self {
        let db_path = "slide_rocksdb".to_string();
        
        // Create RocksDB with optimized settings for SLIDE workload
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_write_buffer_size(64 * 1024 * 1024); // 64MB write buffer
        opts.set_max_write_buffer_number(3);
        opts.set_min_write_buffer_number_to_merge(2);
        opts.set_level_zero_file_num_compaction_trigger(10);
        opts.set_level_zero_slowdown_writes_trigger(20);
        opts.set_level_zero_stop_writes_trigger(40);
        opts.set_max_bytes_for_level_base(512 * 1024 * 1024); // 512MB
        opts.set_max_background_jobs(2);
        opts.set_manual_wal_flush(true);
        opts.set_allow_mmap_reads(true);
        opts.set_allow_mmap_writes(true);
        
        let db = DB::open(&opts, &db_path)
            .unwrap_or_else(|e| panic!("Failed to open RocksDB at {}: {}", db_path, e));

        RocksKVS {
            db,
            db_path,
            cache: HashMap::new(),
        }
    }

    fn read_key(&mut self, key: &str) -> Option<kv_val_t> {
        // Check cache first
        if let Some(val) = self.cache.get(key) {
            return Some(val.clone());
        }

        // Read from RocksDB
        match self.db.get(key.as_bytes()) {
            Ok(Some(data)) => {
                match bincode::deserialize::<kv_val_t>(&data) {
                    Ok(val) => {
                        // Cache the result
                        self.cache.insert(key.to_string(), val.clone());
                        Some(val)
                    }
                    Err(_) => None,
                }
            }
            Ok(None) => None,
            Err(_) => None,
        }
    }

    fn write_kv(&mut self, key: &str, val: kv_val_t) {
        // Update cache
        self.cache.insert(key.to_string(), val.clone());
        
        // Write to RocksDB with optimized settings
        let mut write_opts = WriteOptions::default();
        write_opts.disable_wal(true); // Disable WAL for better performance
        
        if let Ok(serialized) = bincode::serialize(&val) {
            let _ = self.db.put_opt(key.as_bytes(), serialized, &write_opts);
        }
    }

    fn clear(&mut self) {
        self.cache.clear();
        // For RocksDB, we'll just recreate the database
        let _ = &self.db; // Keep reference to avoid dropping
        let _ = DB::destroy(&Options::default(), &self.db_path);
        *self = Self::new();
    }
    
    fn len(&self) -> usize {
        self.cache.len() // Approximate count from cache
    }

    fn flush(&self) {
        let _ = self.db.flush();
    }
}

// Global RocksDB KVS instance
lazy_static::lazy_static! {
    static ref GLOBAL_KVS: Arc<Mutex<RocksKVS>> = Arc::new(Mutex::new(RocksKVS::new()));
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum block_storage_status {
    RejectOldBlock = 1,
    RejectDc = 2,
    RejectTxFormatError = 3,
    Accept = 4,
}

// Simple context management for compatibility
pub fn new_context(_tid: i32) -> i32 {
    static mut NEXT_ID: i32 = 1;
    unsafe {
        let id = NEXT_ID;
        NEXT_ID += 1;
        id
    }
}


// Direct KVS operations using RocksDB implementation
pub fn read_key(_ctx: i32, key: &kv_key_t) -> Box<kv_val_t> {
    let mut kvs = GLOBAL_KVS.lock().unwrap();
    if let Some(val) = kvs.read_key(key) {
        Box::new(val)
    } else {
        Box::new(kv_val_t {
            data: vec![],
            dtype: kv_val_datatype_t::NOT_FOUND,
            ts: 0,
        })
    }
}

pub fn write_kv(_ctx: i32, key: &kv_key_t, val: &kv_val_t) {
    let mut kvs = GLOBAL_KVS.lock().unwrap();
    kvs.write_kv(key, val.clone());
}

pub fn commit_tx(_ctx: i32) -> block_storage_status {
    block_storage_status::Accept
}

pub fn clear_global_kvs() {
    let mut kvs = GLOBAL_KVS.lock().unwrap();
    kvs.clear();
}

pub fn get_global_kvs_size() -> usize {
    let kvs = GLOBAL_KVS.lock().unwrap();
    kvs.len()
}

pub fn flush_global_kvs() {
    let kvs = GLOBAL_KVS.lock().unwrap();
    kvs.flush();
}

// Stub functions for compatibility
pub fn add_lock(_ctx: i32, _name: &kv_key_t) {}
pub fn add_lock_group(_ctx: i32, _names: &[kv_key_t]) {}
pub fn send_datagram(_data: &str, _client: &str, _ty: u64) {}
pub fn barrier(tid: i32, _barrier_name: &str, _num_wait: i32) {
    let _ctx = new_context(tid);
}
pub fn multicast_block(_buf: &[u8], _blk_hsh_out: &mut [u8]) -> block_storage_status {
    block_storage_status::Accept
}

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::LazyLock;

// File operations with proper implementation
static FILE_HANDLES: LazyLock<Mutex<HashMap<u32, BufReader<File>>>> = LazyLock::new(|| Mutex::new(HashMap::new()));
static mut NEXT_FILE_ID: u32 = 1;

pub fn open_file(path: &str) -> u32 {
    match File::open(path) {
        Ok(file) => {
            let reader = BufReader::new(file);
            unsafe {
                let id = NEXT_FILE_ID;
                NEXT_FILE_ID += 1;
                let mut handles = FILE_HANDLES.lock().unwrap();
                handles.insert(id, reader);
                id
            }
        },
        Err(e) => {
            eprintln!("Failed to open file {}: {}", path, e);
            0 // Return 0 to indicate failure
        }
    }
}

pub fn read_file(fd: u32, buf: &mut [std::os::raw::c_char]) -> usize {
    let mut handles = FILE_HANDLES.lock().unwrap();
    if let Some(reader) = handles.get_mut(&fd) {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => 0, // EOF
            Ok(_) => {
                // Remove trailing newline/carriage return
                let trimmed = line.trim_end_matches(&['\r', '\n'][..]);
                let bytes = trimmed.as_bytes();
                let copy_len = bytes.len().min(buf.len());
                for (i, &byte) in bytes.iter().take(copy_len).enumerate() {
                    buf[i] = byte as std::os::raw::c_char;
                }
                // Add null terminator if there's room
                if copy_len < buf.len() {
                    buf[copy_len] = 0;
                }
                copy_len
            },
            Err(_) => 0, // Error, treat as EOF
        }
    } else {
        0 // Invalid file descriptor
    }
}

pub fn close_file(fd: u32) {
    let mut handles = FILE_HANDLES.lock().unwrap();
    handles.remove(&fd);
}

pub fn get_open_files_count() -> usize {
    let handles = FILE_HANDLES.lock().unwrap();
    handles.len()
}

// Simplified async module for compatibility
pub mod psl_async {
    use super::block_storage_status;

    pub struct PromiseInt {
        val: i32,
    }

    pub struct PromiseVoid;

    pub fn commit(_ctx: i32) -> PromiseInt {
        PromiseInt { val: block_storage_status::Accept as i32 }
    }

    impl PromiseInt {
        pub fn wait(self) -> i32 {
            self.val
        }
    }

    pub fn abort(_ctx: i32) -> PromiseVoid {
        PromiseVoid
    }

    impl PromiseVoid {
        pub fn wait(self) {
            // No-op for dummy implementation
        }
    }

    pub fn generic_async_fn(_fn_name: &str) -> PromiseVoid {
        PromiseVoid
    }
}
