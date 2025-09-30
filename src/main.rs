mod bucket;
mod cnpy;
mod config;
mod densified_min_hash;
mod densified_wta_hash;
mod layer;
mod lsh;
mod murmurhash;
mod network;
mod node;
mod psl_array;
mod scl;
mod srp;
mod types;
mod wta_hash;
use std::env;
use std::time::Instant;

use crate::config::*;
use crate::network::Network;
use crate::node::NodeType;

use crate::scl::{open_file as openFile, read_file as ReadFile};

const MAX_LINE_SIZE: usize = 38_000;

fn trim(s: &str) -> String {
    let s = s.trim();
    s.to_string()
}

#[allow(non_snake_case)]
#[derive(Default)]
struct ParsedCfg {
    RangePow: Vec<usize>,
    K: Vec<usize>,
    L: Vec<usize>,
    Sparsity: Vec<f32>,
    Batchsize: usize,
    Rehash: usize,
    Rebuild: usize,
    InputDim: usize,
    totRecords: usize,
    totRecordsTest: usize,
    Lr: f32,
    Epoch: usize,
    Stepsize: usize,
    sizesOfLayers: Vec<usize>,
    numLayer: usize,
    trainData: String,
    testData: String,
    Weights: String,
    savedWeights: String,
    logFile: String,
}

fn parseconfig(content: &str, out: &mut ParsedCfg) {
    for raw in content.lines() {
        let line = raw.trim();
        if line.is_empty() { continue; }
        if line.starts_with('#') { continue; }
        if line.len() < 3 { continue; }

        let Some(eq) = line.find('=') else { 
            eprintln!("Error Parsing conf File at Line\n{}", line);
            continue;
        };
        let first = trim(&line[..eq]);
        let second = trim(&line[eq + 1..]);

        let parse_list_usize = |s: &str| -> Vec<usize> {
            s.split(',')
                .map(|t| t.trim())
                .filter(|t| !t.is_empty())
                .map(|t| t.parse::<usize>().unwrap_or(0))
                .collect()
        };
        let parse_list_f32 = |s: &str| -> Vec<f32> {
            s.split(',')
                .map(|t| t.trim())
                .filter(|t| !t.is_empty())
                .map(|t| t.parse::<f32>().unwrap_or(0.0))
                .collect()
        };

        match first.as_str() {
            "RangePow" => out.RangePow = parse_list_usize(&second),
            "K"        => out.K        = parse_list_usize(&second),
            "L"        => out.L        = parse_list_usize(&second),
            "Sparsity" => out.Sparsity = parse_list_f32(&second),

            "Batchsize"      => out.Batchsize = second.parse().unwrap_or(out.Batchsize),
            "Rehash"         => out.Rehash = second.parse().unwrap_or(out.Rehash),
            "Rebuild"        => out.Rebuild = second.parse().unwrap_or(out.Rebuild),
            "InputDim"       => out.InputDim = second.parse().unwrap_or(out.InputDim),
            "totRecords"     => out.totRecords = second.parse().unwrap_or(out.totRecords),
            "totRecordsTest" => out.totRecordsTest = second.parse().unwrap_or(out.totRecordsTest),
            "Epoch"          => out.Epoch = second.parse().unwrap_or(out.Epoch),
            "Lr"             => out.Lr = second.parse().unwrap_or(out.Lr),
            "Stepsize"       => out.Stepsize = second.parse().unwrap_or(out.Stepsize),
            "numLayer"       => out.numLayer = second.parse().unwrap_or(out.numLayer),
            "logFile"        => out.logFile = second.to_string(),

            "sizesOfLayers"  => out.sizesOfLayers = parse_list_usize(&second),
            "trainData"      => out.trainData = second.to_string(),
            "testData"       => out.testData = second.to_string(),
            "weight"         => out.Weights = second.to_string(),
            "savedweight"    => out.savedWeights = second.to_string(),

            _ => {
                eprintln!("Error Parsing conf File at Line");
                eprintln!("{}", line);
            }
        }
    }
}

fn parse_number(bin_name: &str) -> i32 {
    let b = bin_name.as_bytes();
    if b.is_empty() || b[0] != b'w' || b.len() == 1 {
        eprintln!("Invalid string format: {}", bin_name);
        return -1;
    }
    let rest = &bin_name[1..];
    match rest.parse::<i32>() {
        Ok(n) => n,
        Err(_) => {
            eprintln!("Error parsing number from string: {}", bin_name);
            -1
        }
    }
}

fn eval_data_svm(num_batches_test: usize, net: &mut Network, iter: i32, test_path: &str, batchsize: usize) {
    let mut tot_correct = 0i32;
    let fd = openFile(test_path);
    let mut line_buf = vec![0i8; MAX_LINE_SIZE];

    let _ = ReadFile(fd, &mut line_buf);

    let mut num_actual_batch_tests = 0usize;

    for i in 0..num_batches_test {
        let mut records: Vec<Vec<usize>> = Vec::with_capacity(batchsize);
        let mut values:  Vec<Vec<f32>>   = Vec::with_capacity(batchsize);
        let mut sizes:   Vec<usize>      = Vec::with_capacity(batchsize);
        let mut labels:  Vec<Vec<usize>> = Vec::with_capacity(batchsize);
        let mut labelsize: Vec<usize>    = Vec::with_capacity(batchsize);

        let mut count = 0usize;
        while let Some(line) = read_line_utf8(fd, &mut line_buf) {
            if line.is_empty() { continue; }

            let (labels_part, feats_part) = match line.split_once(' ') {
                Some((lp, fp)) => (lp, fp),
                None => (line.as_str(), ""),
            };

            let cur_labels: Vec<usize> = labels_part
                .split(',')
                .filter(|t| !t.is_empty())
                .filter_map(|t| t.parse::<usize>().ok())
                .collect();

            let mut cur_idx = Vec::<usize>::new();
            let mut cur_val = Vec::<f32>::new();
            if !feats_part.is_empty() {
                for tok in feats_part.split_whitespace() {
                    if let Some((i_str, v_str)) = tok.split_once(':') {
                        if let (Ok(ii), Ok(vv)) = (i_str.parse::<usize>(), v_str.parse::<f32>()) {
                            cur_idx.push(ii);
                            cur_val.push(vv);
                        }
                    }
                }
            }

            sizes.push(cur_idx.len());
            labelsize.push(cur_labels.len());
            records.push(cur_idx);
            values.push(cur_val);
            labels.push(cur_labels);

            count += 1;
            if count >= batchsize { break; }
        }

        let num_features: usize = sizes.iter().sum();
        let num_labels: usize = labelsize.iter().sum();

        if num_labels != 0 {
            num_actual_batch_tests += 1;

            println!(
                "{} records, with {} features and {} labels",
                batchsize, num_features, num_labels
            );

            let rec_slices: Vec<Vec<usize>> = records.into_iter().collect();
            let val_slices: Vec<Vec<f32>>   = values.into_iter().collect();
            let lbl_slices: Vec<Vec<usize>> = labels.into_iter().collect();

            let correct = net.predict_class(
                &rec_slices,
                &val_slices,
                &sizes,
                &lbl_slices,
                &labelsize,
                count,
            );
            tot_correct += correct;
            println!(
                " iter {}: {} correct with totCorrect {}",
                i,
                (tot_correct as f64) / ((batchsize * (i + 1)) as f64),
                tot_correct
            );
        }
    }

    println!(
        "over all {}",
        (tot_correct as f64) / ((num_actual_batch_tests * batchsize) as f64)
    );
    println!(
        "{} {} {}",
        iter,
        0,
        (tot_correct as f64) / ((num_actual_batch_tests * batchsize) as f64)
    );
}

fn read_data_svm(num_batches: usize, net: &mut Network, epoch: usize, cfg: &ParsedCfg) {
    let fd = openFile(&cfg.trainData);
    let mut line_buf = vec![0i8; MAX_LINE_SIZE];

    let _ = ReadFile(fd, &mut line_buf);

    for i in 0..num_batches {
        if ((i + epoch * num_batches) % cfg.Stepsize) == 0 {
            eval_data_svm(20, net, (epoch * num_batches + i) as i32, &cfg.testData, cfg.Batchsize);
        }
        
        // Debug: Show progress every 10 batches
        if i % 10 == 0 {
            println!("[DEBUG] Processing batch {}/{} in epoch {}", i, num_batches, epoch);
        }
        
        // Show weight statistics after first few batches to confirm updates
        if i == 5 && epoch == 0 {
            println!("=== WEIGHTS AFTER 5 BATCHES ===");
            net.report_weight_stats();
        }

        let mut records: Vec<Vec<usize>> = Vec::with_capacity(cfg.Batchsize);
        let mut values:  Vec<Vec<f32>>   = Vec::with_capacity(cfg.Batchsize);
        let mut sizes:   Vec<usize>      = Vec::with_capacity(cfg.Batchsize);
        let mut labels:  Vec<Vec<usize>> = Vec::with_capacity(cfg.Batchsize);
        let mut labelsize: Vec<usize>    = Vec::with_capacity(cfg.Batchsize);

        let mut count = 0usize;
        while let Some(line) = read_line_utf8(fd, &mut line_buf) {
            if line.is_empty() { continue; }

            let (labels_part, feats_part) = match line.split_once(' ') {
                Some((lp, fp)) => (lp, fp),
                None => (line.as_str(), ""),
            };

            let cur_labels: Vec<usize> = labels_part
                .split(',')
                .filter(|t| !t.is_empty())
                .filter_map(|t| t.parse::<usize>().ok())
                .collect();

            let mut cur_idx = Vec::<usize>::new();
            let mut cur_val = Vec::<f32>::new();
            if !feats_part.is_empty() {
                for tok in feats_part.split_whitespace() {
                    if let Some((i_str, v_str)) = tok.split_once(':') {
                        if let (Ok(ii), Ok(vv)) = (i_str.parse::<usize>(), v_str.parse::<f32>()) {
                            cur_idx.push(ii);
                            cur_val.push(vv);
                        }
                    }
                }
            }

            sizes.push(cur_idx.len());
            labelsize.push(cur_labels.len());
            records.push(cur_idx);
            values.push(cur_val);
            labels.push(cur_labels);

            count += 1;
            if count >= cfg.Batchsize { break; }
        }

        let mut rehash = false;
        let mut rebuild = false;

        if cfg.Rehash > 0 && cfg.Batchsize > 0 {
            if ((epoch * num_batches + i) % (cfg.Rehash / cfg.Batchsize)) == (cfg.Rehash / cfg.Batchsize - 1) {
                if Mode == 1 || Mode == 4 {
                    rehash = true;
                }
            }
        }
        if cfg.Rebuild > 0 && cfg.Batchsize > 0 {
            if ((epoch * num_batches + i) % (cfg.Rebuild / cfg.Batchsize)) == (cfg.Rehash / cfg.Batchsize - 1) {
                if Mode == 1 || Mode == 4 {
                    rebuild = true;
                }
            }
        }

        let t1 = Instant::now();
        let rec_slices: Vec<Vec<usize>> = records;
        let val_slices: Vec<Vec<f32>>   = values;
        let lbl_slices: Vec<Vec<usize>> = labels;

        let _ = net.process_input(
            &rec_slices,
            &val_slices,
            &sizes,
            &lbl_slices,
            &labelsize,
            (epoch * num_batches + i) as i32,
            rehash,
            rebuild,
        );
        let t2 = Instant::now();
        let _time_ms = (t2 - t1).as_millis() as i64;
    }
}

fn default_config_blob() -> String {
    r#"
            RangePow = 6,18
            K = 2,6
            L = 20,50
            Sparsity = 1,0.005,1,1

            Batchsize=128
            Rehash=6400
            Rebuild=128000
            InputDim=5000
            totRecords=15539
            totRecordsTest=3809

            Lr=0.0001
            Epoch=10
            Stepsize=1000

            sizesOfLayers=128,3993
            numLayer=2

            trainData=/app_configs/Eurlex/eurlex_train.txt
            testData=/app_configs/Eurlex/eurlex_test.txt

            weight=../savedWeight.npz
            savedweight=../savedWeight.npz

            logFile=../dataset/log.txt
    "#
    .to_string()
}

fn read_line_utf8(fd: u32, buf: &mut [i8]) -> Option<String> {
    let n = ReadFile(fd, buf);
    if n == 0 { return None; }
    let bytes: Vec<u8> = buf[..n].iter().map(|&b| b as u8).collect();
    let s = std::str::from_utf8(&bytes).ok()?
        .trim_end_matches(char::from(0))
        .trim()
        .to_string();
    Some(s)
}


fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <test_hash> <train_hash>", args.get(0).unwrap_or(&"w1".to_string()));
        return;
    }

    println!("Starting Deep Learning Benchmark as worker: {}", args[0]);
    let worker_id = if args[0].contains("slide-in-rust") {
        0 // Default to worker 0 for single-worker mode
    } else {
        parse_number(&args[0]) - 1
    };

    let hash_test  = args[1].clone();
    let hash_train = args[2].clone();

    println!("Training hash: {}", hash_train);
    println!("Test hash: {}", hash_test);

    let mut cfg = ParsedCfg {
        Batchsize: 32,   // Much smaller batch size
        Rehash: 1000,
        Rebuild: 1000,
        InputDim: 784,
        totRecords: 320,  // Very small for testing - only 10 batches
        totRecordsTest: 128,  // Very small for testing
        Lr: 0.0001,
        Epoch: 1,  // Just one epoch for testing
        Stepsize: 20,
        numLayer: 2,  // Reduce to 2 layers
        ..Default::default()
    };
    let blob = default_config_blob();
    parseconfig(&blob, &mut cfg);

    cfg.trainData = hash_train;
    cfg.testData = hash_test;
    
    // Override for testing with smaller values - AFTER parseconfig to ensure they take effect
    cfg.sizesOfLayers = vec![32, 100];  // Small network: 32 hidden nodes, 100 output classes
    cfg.totRecords = 320;  // Very small for testing - only 10 batches
    cfg.totRecordsTest = 128;  // Very small for testing
    cfg.Batchsize = 32;   // Much smaller batch size
    cfg.Epoch = 1;  // Just one epoch for testing
    cfg.numLayer = 2;  // Reduce to 2 layers
    // Keep the original InputDim to match the Amazon dataset
    cfg.InputDim = 203882;  // Match Amazon dataset feature space

    let num_batches = cfg.totRecords / cfg.Batchsize;
    let num_batches_test = cfg.totRecordsTest / cfg.Batchsize;
    println!("numBatches: {}, numBatchesTest:{}", num_batches, num_batches_test);

    let mut layers_types = vec![NodeType::ReLU; cfg.numLayer.max(1)];
    if !layers_types.is_empty() {
        layers_types[cfg.numLayer - 1] = NodeType::Softmax;
    }

    let t1 = Instant::now();
    let mut net = Network::new(
        cfg.sizesOfLayers.clone(),
        layers_types,
        cfg.numLayer,
        cfg.Batchsize,
        cfg.Lr,
        cfg.InputDim,
        &cfg.K,
        &cfg.L,
        &cfg.RangePow,
        cfg.Sparsity.clone(),
        (),
        worker_id as usize,
    );
    let t2 = Instant::now();
    let time_ms = (t2 - t1).as_micros() as f64 / 1000.0;
    println!("Network Initialization takes {} milliseconds", time_ms);

    // Report initial weight statistics
    println!("=== INITIAL WEIGHTS ===");
    net.report_weight_stats();

    for e in 0..cfg.Epoch {
        println!("=== STARTING EPOCH {} ===", e);
        read_data_svm(num_batches, &mut net, e, &cfg);

        // Report weight statistics after each epoch  
        println!("=== WEIGHTS AFTER EPOCH {} ===", e);
        net.report_weight_stats();

        if e == cfg.Epoch - 1 {
            eval_data_svm(num_batches_test, &mut net, ((e + 1) * num_batches) as i32, &cfg.testData, cfg.Batchsize);
        } else {
            eval_data_svm(50, &mut net, ((e + 1) * num_batches) as i32, &cfg.testData, cfg.Batchsize);
        }
        net.save_weights(&cfg.savedWeights);
    }
}
