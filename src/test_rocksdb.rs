// Simple test to verify RocksDB KVS functionality
use std::process;

mod types;
mod scl;
mod config;

use types::{kv_val_t, kv_val_datatype_t};
use scl::{new_context, read_key, write_kv, commit_tx};

fn main() {
    println!("Testing RocksDB KVS Integration...");

    // Test basic write/read operations
    let ctx = new_context(1);
    let test_key = "test_key_123".to_string();
    let test_data = vec![1u8, 2u8, 3u8, 4u8, 5u8];
    
    let test_val = kv_val_t {
        data: test_data.clone(),
        dtype: kv_val_datatype_t::BYTES,
        ts: 123456,
    };

    println!("Writing test data to RocksDB...");
    write_kv(ctx, &test_key, &test_val);
    let _ = commit_tx(ctx);

    println!("Reading test data from RocksDB...");
    let ctx2 = new_context(2);
    let retrieved_val = read_key(ctx2, &test_key);

    if retrieved_val.dtype == kv_val_datatype_t::NOT_FOUND {
        println!("❌ ERROR: Key not found in RocksDB!");
        process::exit(1);
    } else {
        println!("✅ SUCCESS: Key found in RocksDB!");
        println!("   Original data: {:?}", test_data);
        println!("   Retrieved data: {:?}", retrieved_val.data);
        println!("   Data matches: {}", test_data == retrieved_val.data);
        println!("   Timestamp: {}", retrieved_val.ts);
    }

    // Test multiple keys
    println!("\nTesting multiple keys...");
    for i in 0..5usize {
        let ctx = new_context((i + 10) as i32);
        let key = format!("multi_key_{}", i);
        let data = vec![i as u8; i + 1];
        let val = kv_val_t {
            data,
            dtype: kv_val_datatype_t::BYTES,
            ts: i,
        };
        write_kv(ctx, &key, &val);
        let _ = commit_tx(ctx);
    }

    // Read back all keys
    for i in 0..5usize {
        let ctx = new_context((i + 20) as i32);
        let key = format!("multi_key_{}", i);
        let retrieved = read_key(ctx, &key);
        if retrieved.dtype != kv_val_datatype_t::NOT_FOUND {
            println!("   Key {}: data length = {}, ts = {}", i, retrieved.data.len(), retrieved.ts);
        } else {
            println!("   ❌ Key {} not found!", i);
        }
    }

    println!("\n🎉 RocksDB KVS integration test completed successfully!");
}