#![allow(non_camel_case_types, non_snake_case, dead_code)]

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum op_t {
    OP_GET    = 1 << 20,
    OP_PUT    = 1 << 21,
    OP_DELETE = 1 << 22,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum resp_t {
    RESP_FOUND     = 1 << 15,
    RESP_NOT_FOUND = 1 << 16,
    RESP_SUCCESS   = 1 << 17,
    RESP_FAILURE   = 1 << 18,
    RESP_MIXED     = 1 << 19,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum dgram_type_t {
    DGRAM_TYPE_ASYNC           = 0,
    DGRAM_TYPE_SYNC            = 1,
    DGRAM_TYPE_QUORUM          = 1 << 1,
    DGRAM_TYPE_SINGLE_RESPONSE = 1 << 2,
    DGRAM_TYPE_DC_REQUEST      = 1 << 3,
    DGRAM_TYPE_DC_RESPONSE     = 1 << 4,
    DGRAM_TYPE_CDB_REQUEST     = 1 << 5,
    DGRAM_TYPE_CDB_RESPONSE    = 1 << 6,
    DGRAM_TYPE_TX_REQUEST      = 1 << 7,
    DGRAM_TYPE_TX_RESPONSE     = 1 << 8,
    DGRAM_TYPE_LOCK_REQUEST    = 1 << 9,
    DGRAM_TYPE_LOCK_RESPONSE   = 1 << 10,
    DGRAM_TYPE_LOCK_RELEASE    = 1 << 11,
    DGRAM_TYPE_WAIT_REQUEST    = 1 << 12,
    DGRAM_TYPE_WAIT_RESPONSE   = 1 << 13,
    DGRAM_TYPE_SYNC_REPORT     = 1 << 14,
    DGRAM_TYPE_ATTESTATION_REPORT = 1 << 23,
    DGRAM_TYPE_KEY_EXCHANGE       = 1 << 24,
    DGRAM_TYPE_ATTESTATION_ACK    = 1 << 25,
    DGRAM_TYPE_APP_LAUNCH         = 1 << 26,
    DGRAM_TYPE_APP_LAUNCH_ACK     = 1 << 27,
    DGRAM_TYPE_APP_FIN            = 1 << 28,
    DGRAM_TYPE_APP_READY          = 1 << 29,
    DGRAM_TYPE_FILE_REQUEST       = 1 << 30,
    DGRAM_TYPE_FILE_RESPONSE      = 1 << 31,
}

pub type kv_key_t = String;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum kv_val_datatype_t {
    NOT_FOUND ,
    DELETED,
    BYTES,
    INTEGER,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct kv_val_t {
    pub data: Vec<u8>,
    pub dtype: kv_val_datatype_t,
    pub ts: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KV_Status {
    READ_PASS,
    WRITE_PASS,
    READ_FAIL,
    WRITE_FAIL,
    JUST_FULL,
    WAIT_FLUSH,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IO_Status {
    IO_READ_FAIL,
    IO_WRITE_FAIL,
    IO_READ_PASS,
    IO_WRITE_PASS,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct host_buf {
    pub buf_data: *mut i8,
    pub buf_meta: *mut i8,
    pub buf_data_size: usize,
    pub buf_meta_size: usize,
}

#[macro_export]
macro_rules! PROTO_TO_RAW_DTYPE {
    ($in:expr, $out:expr) => {{
        if ($in).dtype() == storage::BYTES   { $out = $crate::types::kv_val_datatype_t::BYTES; }
        if ($in).dtype() == storage::INTEGER { $out = $crate::types::kv_val_datatype_t::INTEGER; }
        if ($in).dtype() == storage::DELETED { $out = $crate::types::kv_val_datatype_t::DELETED; }
    }};
}

#[macro_export]
macro_rules! RAW_TO_PROTO_DTYPE {
    ($in:expr, $out:expr) => {{
        if ($in) == $crate::types::kv_val_datatype_t::BYTES   { $out.set_dtype(storage::BYTES); }
        if ($in) == $crate::types::kv_val_datatype_t::INTEGER { $out.set_dtype(storage::INTEGER); }
        if ($in) == $crate::types::kv_val_datatype_t::DELETED { $out.set_dtype(storage::DELETED); }
    }};
}
