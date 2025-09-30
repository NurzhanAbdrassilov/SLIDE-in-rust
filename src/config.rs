pub const ADAM: bool = true;
pub const BETA1: f32 = 0.9;
pub const BETA2: f32 = 0.999;
pub const EPS: f32 = 0.00000001;

pub const NUM_WAIT: i32 = 4;
pub const TID: i32 = 2;

//1: wta; 2: Densified wta; 3: topk minhash; 4: simhash
pub const HASH_FUNCTION_WTA: i32 =       1;
pub const HASH_FUNCTION_DWTA: i32 =      2;
pub const HASH_FUNCTION_TOPK_MIN: i32 =  3;
pub const HASH_FUNCTION_SIMHASH: i32 =   4;

pub const HashFunction: i32 = 2;
pub const BUCKETSIZE: usize = 128;
//for minhash
pub const TOPK: i32 = 30;
//for simhash
pub const Ratio: i32 = 3;
//for wta/dwta
pub const binsize: usize = 8;

//Mode 1: Topk thresholding Mode 4: Sampling
pub const MODE_TOPK_THRESHOLD: i32 =   1;
pub const MODE_SAMPLING: i32 =         4;

pub const Mode: i32 = 4;

pub const THRESH: i32 = 2;

pub const FIFO: bool = true;

pub const LOADWEIGHT: bool = false;

pub const MAPLEN: usize = 325056;