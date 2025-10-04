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

/// Evaluates model performance on test data
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

/// Processes training data in batches for one epoch
fn read_data_svm(num_batches: usize, net: &mut Network, epoch: usize, cfg: &ParsedCfg) {
    let fd = openFile(&cfg.trainData);
    let mut line_buf = vec![0i8; MAX_LINE_SIZE];

    let _ = ReadFile(fd, &mut line_buf);

    for i in 0..num_batches {
        // Evaluate less frequently to avoid hanging - only every 5000 batches instead of every 1000
        if ((i + epoch * num_batches) % (cfg.Stepsize * 5)) == 0 && i > 0 {
            println!("Running evaluation...");
            eval_data_svm(5, net, (epoch * num_batches + i) as i32, &cfg.testData, cfg.Batchsize);
        }
        
        if i % 100 == 0 {
            println!("Batch {}/{} in epoch {} ({:.1}%)", 
                i, num_batches, epoch, (i as f64 / num_batches as f64) * 100.0);
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
        
        // Check weight updates after first few batches
        if (i == 1 || i == 5 || i == 10) && epoch == 0 {
            println!("=== WEIGHT CHECK AFTER BATCH {} ===", i);
            net.report_weight_stats();
            println!("==============================");
        }
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


/// SLIDE (Sub-Linear Deep Learning Engine) implementation for sparse neural networks
/// Trains on the Eurlex-4.3K multilabel text classification dataset
fn main() {
    let args: Vec<String> = env::args().collect();
    
    println!("Starting SLIDE Deep Learning with Eurlex Dataset");
    
    // Parse worker ID from command line arguments  
    let worker_id = if args.len() > 0 && !args[0].contains("slide-in-rust") {
        parse_number(&args[0]) - 1
    } else {
        0 // Single worker mode
    };

    println!("Using Eurlex dataset with {} workers", NUM_WAIT);
    println!("Worker ID: {}", worker_id);

    let mut cfg = ParsedCfg {
        Batchsize: 32,  // Smaller batch size to reduce memory usage
        Rehash: 6400,
        Rebuild: 128000,
        InputDim: 200000,  // Actual Eurlex input dimension
        totRecords: 45000,  // Actual Eurlex training records
        totRecordsTest: 6000,  // Actual Eurlex test records
        Lr: 0.05,  // Increased learning rate for better convergence
        Epoch: 5,  // More epochs for proper training
        Stepsize: 1000,
        numLayer: 2,
        ..Default::default()
    };
    let blob = default_config_blob();
    parseconfig(&blob, &mut cfg);

    cfg.trainData = "dataset\\EURLex-4.3K\\train.txt".to_string();
    cfg.testData = "dataset\\EURLex-4.3K\\test.txt".to_string();
    cfg.sizesOfLayers = vec![64, 4271];
    cfg.InputDim = 200000;
    cfg.totRecords = 45000;
    cfg.totRecordsTest = 6000;
    cfg.Batchsize = 32;
    cfg.Epoch = 5;
    cfg.numLayer = 2;
    cfg.Lr = 0.01;

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
    
    // Display training configuration
    println!("Configuration: LR={}, Batch={}, Input Dim={}, Layers={:?}", 
        cfg.Lr, cfg.Batchsize, cfg.InputDim, cfg.sizesOfLayers);
    
    println!("Initial network statistics:");
    net.report_weight_stats();

    // Main training loop
    for e in 0..cfg.Epoch {
        println!("Starting epoch {}/{}", e + 1, cfg.Epoch);
        
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            read_data_svm(num_batches, &mut net, e, &cfg);
        })).unwrap_or_else(|_| {
            println!("ERROR: Training failed at epoch {}", e);
            std::process::exit(1);
        });

        println!("Completed epoch {}", e + 1);
        net.report_weight_stats();

        // Evaluation
        if e == cfg.Epoch - 1 {
            eval_data_svm(num_batches_test, &mut net, ((e + 1) * num_batches) as i32, &cfg.testData, cfg.Batchsize);
        } else {
            eval_data_svm(50, &mut net, ((e + 1) * num_batches) as i32, &cfg.testData, cfg.Batchsize);
        }
        
        net.save_weights(&cfg.savedWeights);
    }
    
    println!("Training completed successfully");
}
