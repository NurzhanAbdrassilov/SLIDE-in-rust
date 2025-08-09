use std::os::raw::{c_char, c_int};
use crate::types::{kv_key_t, kv_val_datatype_t, kv_val_t};

pub const MAX_VAL_SIZE: usize = std::mem::size_of::<f32>() * (1 << 20);

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum block_storage_status {
    REJECT_OLD_BLOCK = 1,
    REJECT_DC = 2,
    REJECT_TX_FORMAT_ERROR = 3,
    ACCEPT = 4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct wasm_kv_key_t {
    pub key: u64,
    pub size: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct wasm_kv_val_t {
    pub data: u64,
    pub size: u64,
    pub dtype: kv_val_datatype_t,
    pub ts: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct wasm_lock_names {
    pub lock_names: u64,
    pub size: u64,
}

pub mod ffi {
    use super::{block_storage_status, wasm_kv_key_t, wasm_kv_val_t};
    use std::os::raw::{c_char, c_int};
    use crate::types::{kv_val_datatype_t};

    #[link(wasm_import_module = "wasi_snapshot_preview1")]
    extern "C" {
        pub fn debug(x: c_int);

        pub fn NewContext(tid: c_int) -> c_int;

        pub fn _openFile(path: *mut c_char, path_size: usize) -> c_int;

        pub fn batchTransactions(buf: *const u8, buf_size: usize);

        pub fn _ReadKey(
            ctx: c_int,
            key: *const wasm_kv_key_t,
            buf: *mut u8,
            buf_size: *mut usize,
            dtype: *mut kv_val_datatype_t,
            ts: *mut usize,
        ) -> c_int;

        pub fn _WriteKV(ctx: c_int, key: *const wasm_kv_key_t, val: wasm_kv_val_t);

        pub fn _AddLock(ctx: c_int, name: *const wasm_kv_key_t);

        pub fn _Barrier(barrier_name: *const u8, barrier_name_sz: usize, num_wait: c_int, ctx: c_int);

        pub fn _AddLockGroup(ctx: c_int, names: *const wasm_kv_key_t, size: c_int);

        pub fn _Commit(ctx: c_int, ret: *mut block_storage_status);

        pub fn Input(input: *mut u8, input_size: *mut usize);

        pub fn _ReadFile(fd: u32, file_buf: *mut c_char, file_size: usize) -> usize;

        pub fn Abort(ctx: c_int);

        pub fn _MulticastBlock(
            buf: *const c_char,
            buf_size: usize,
            blk_hsh: *mut c_char,
            ret: *mut block_storage_status,
        );

        pub fn DownloadBlockDataFromDC(hsh: *const c_char, buf: *mut c_char, size: *mut c_int, max_sz: c_int);

        pub fn retrieveBatch(tx_buf: *mut u8, tx_buf_sz: *mut usize);

        pub fn _sentDatagram(
            txr_buf: *const u8,
            txr_buf_sz: usize,
            client_addr: *const u8,
            client_size: usize,
            r#type: u64,
        );

        pub fn RegisterResponse(cnt: u64);

        pub fn AsyncFn_1int(
            idx: c_int,
            promise_num: u64,
            fn_name: *const c_char,
            fn_name_sz: usize,
            input: u64,
        );

        pub fn AsyncReturn_int(idx: c_int, input: *mut c_int);

        pub fn Async_GetMode(idx: c_int) -> c_int;

        pub fn Async_GetPromiseNum(idx: c_int) -> u64;

        pub fn Async_ResetMode(idx: c_int);

        pub fn GetCtxRawPtr(idx: c_int) -> u64;
    }
}


pub fn open_file(path: &str) -> u32 {
    let bytes = path.as_bytes();
    unsafe { ffi::_openFile(bytes.as_ptr() as *mut c_char, bytes.len()) as u32 }
}

pub fn read_file(fd: u32, buf: &mut [c_char]) -> usize {
    unsafe { ffi::_ReadFile(fd, buf.as_mut_ptr(), buf.len()) }
}

pub fn read_key(ctx: c_int, key: &kv_key_t) -> Box<kv_val_t> {
    let mut data = vec![0u8; MAX_VAL_SIZE];
    let mut val_size: usize = MAX_VAL_SIZE;
    let mut dtype = kv_val_datatype_t::NOT_FOUND;
    let mut ts: usize = 0;

    let kb = key.as_bytes();
    let k = wasm_kv_key_t { key: kb.as_ptr() as u64, size: kb.len() as u64 };

    unsafe {
        ffi::_ReadKey(ctx, &k, data.as_mut_ptr(), &mut val_size, &mut dtype, &mut ts);
    }
    assert!(val_size <= MAX_VAL_SIZE);
    data.truncate(val_size);

    Box::new(kv_val_t { data, dtype, ts })
}

pub fn write_kv(ctx: c_int, key: &kv_key_t, val: &kv_val_t) {
    let kb = key.as_bytes();
    let k = wasm_kv_key_t { key: kb.as_ptr() as u64, size: kb.len() as u64 };
    let v = wasm_kv_val_t {
        data: val.data.as_ptr() as u64,
        size: val.data.len() as u64,
        dtype: kv_val_datatype_t::BYTES,
        ts: 0,
    };

    unsafe { ffi::_WriteKV(ctx, &k, v) }
}

pub fn add_lock(ctx: c_int, name: &kv_key_t) {
    let nb = name.as_bytes();
    let k = wasm_kv_key_t { key: nb.as_ptr() as u64, size: nb.len() as u64 };
    unsafe { ffi::_AddLock(ctx, &k) }
}

pub fn add_lock_group(ctx: c_int, names: &[kv_key_t]) {
    let temp: Vec<wasm_kv_key_t> = names.iter().map(|s| {
        let b = s.as_bytes();
        wasm_kv_key_t { key: b.as_ptr() as u64, size: b.len() as u64 }
    }).collect();

    unsafe { ffi::_AddLockGroup(ctx, temp.as_ptr(), temp.len() as c_int) }
}

pub fn commit_tx(ctx: c_int) -> block_storage_status {
    let mut ret = block_storage_status::REJECT_OLD_BLOCK;
    unsafe { ffi::_Commit(ctx, &mut ret) };
    ret
}

pub fn send_datagram(data: &str, client: &str, ty: u64) {
    let db = data.as_bytes();
    let cb = client.as_bytes();
    unsafe { ffi::_sentDatagram(db.as_ptr(), db.len(), cb.as_ptr(), cb.len(), ty) }
}

pub fn barrier(tid: c_int, barrier_name: &str, num_wait: c_int) {
    let nb = barrier_name.as_bytes();
    unsafe {
        let ctx = ffi::NewContext(tid);
        ffi::_Barrier(nb.as_ptr(), nb.len(), num_wait, ctx);
        ffi::Abort(ctx);
    }
}

pub fn multicast_block(buf: &[u8], blk_hsh_out: &mut [u8]) -> block_storage_status {
    let mut ret = block_storage_status::REJECT_OLD_BLOCK;
    unsafe {
        ffi::_MulticastBlock(
            buf.as_ptr() as *const c_char,
            buf.len(),
            blk_hsh_out.as_mut_ptr() as *mut c_char,
            &mut ret,
        );
    }
    ret
}

pub mod PSLAsync {
    use super::ffi;
    use super::block_storage_status;
    use std::os::raw::c_int;

    pub struct PromiseInt {
        runner: c_int,
        ready: bool,
        val: c_int,
    }

    pub struct PromiseVoid {
        runner: c_int,
        ready: bool,
    }

    fn find_idle_runner() -> Option<c_int> {
        unsafe {
            for i in 0..2 {
                if ffi::Async_GetMode(i) == 0 {
                    return Some(i);
                }
            }
            None
        }
    }

    pub fn commit(ctx: c_int) -> PromiseInt {
        unsafe {
            if let Some(runner) = find_idle_runner() {
                let pnum = ffi::Async_GetPromiseNum(runner);
                let name = b"Commit";
                let ctx_ptr = ffi::GetCtxRawPtr(ctx);
                ffi::AsyncFn_1int(
                    runner,
                    pnum,
                    name.as_ptr() as _,
                    name.len(),
                    ctx_ptr,
                );
                PromiseInt { runner, ready: false, val: 0 }
            } else {
                let mut ret = block_storage_status::REJECT_OLD_BLOCK;
                ffi::_Commit(ctx, &mut ret);
                PromiseInt { runner: -1, ready: true, val: ret as c_int }
            }
        }
    }

    impl PromiseInt {
        pub fn wait(self) -> i32 {
            unsafe {
                if self.ready {
                    return self.val;
                }
                while ffi::Async_GetMode(self.runner) != 3 {}
                let mut out: c_int = 0;
                ffi::AsyncReturn_int(self.runner, &mut out);
                ffi::Async_ResetMode(self.runner);
                out
            }
        }
    }

    pub fn abort(ctx: c_int) -> PromiseVoid {
        unsafe {
            if let Some(runner) = find_idle_runner() {
                let pnum = ffi::Async_GetPromiseNum(runner);
                let name = b"Abort";
                let ctx_ptr = ffi::GetCtxRawPtr(ctx);
                ffi::AsyncFn_1int(
                    runner,
                    pnum,
                    name.as_ptr() as _,
                    name.len(),
                    ctx_ptr,
                );
                PromiseVoid { runner, ready: false }
            } else {
                ffi::Abort(ctx);
                PromiseVoid { runner: -1, ready: true }
            }
        }
    }

    impl PromiseVoid {
        pub fn wait(self) {
            unsafe {
                if self.ready {
                    return;
                }
                while ffi::Async_GetMode(self.runner) != 3 {}
                ffi::Async_ResetMode(self.runner);
            }
        }
    }

    pub fn generic_async_fn(fn_name: &str) -> PromiseVoid {
        unsafe {
            if let Some(runner) = find_idle_runner() {
                let pnum = ffi::Async_GetPromiseNum(runner);
                ffi::AsyncFn_1int(
                    runner,
                    pnum,
                    fn_name.as_ptr() as _,
                    fn_name.len(),
                    0,
                );
                PromiseVoid { runner, ready: false }
            } else {
                PromiseVoid { runner: -1, ready: true }
            }
        }
    }
}
