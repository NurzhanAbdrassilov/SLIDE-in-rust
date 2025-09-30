use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use crate::types::{kv_key_t, kv_val_datatype_t, kv_val_t};

pub const MAX_VAL_SIZE: usize = std::mem::size_of::<f32>() * (1 << 20);

// Dummy KVS implementation using HashMap
struct DummyKVS {
    storage: HashMap<String, kv_val_t>,
}

impl DummyKVS {
    fn new() -> Self {
        DummyKVS {
            storage: HashMap::new(),
        }
    }

    fn read_key(&self, key: &str) -> Option<kv_val_t> {
        self.storage.get(key).cloned()
    }

    fn write_kv(&mut self, key: &str, val: kv_val_t) {
        self.storage.insert(key.to_string(), val);
    }

    #[allow(dead_code)]
    fn clear(&mut self) {
        self.storage.clear();
    }
}

// Global dummy KVS instance
lazy_static::lazy_static! {
    static ref GLOBAL_KVS: Arc<Mutex<DummyKVS>> = Arc::new(Mutex::new(DummyKVS::new()));
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


// Direct KVS operations using the dummy HashMap implementation
pub fn read_key(_ctx: i32, key: &kv_key_t) -> Box<kv_val_t> {
    let kvs = GLOBAL_KVS.lock().unwrap();
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
                let bytes = line.as_bytes();
                let copy_len = bytes.len().min(buf.len());
                for (i, &byte) in bytes.iter().take(copy_len).enumerate() {
                    buf[i] = byte as std::os::raw::c_char;
                }
                copy_len
            },
            Err(_) => 0, // Error, treat as EOF
        }
    } else {
        0 // Invalid file descriptor
    }
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
