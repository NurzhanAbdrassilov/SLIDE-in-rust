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

use crate::config::*;
use crate::network::Network;
use crate::node::NodeType;

use crate::scl::{open_file as openFile, read_file as ReadFile, close_file as closeFile};

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
    let total_samples = num_batches_test * batchsize;
    println!("Evaluation: {} batches × {} batch_size = {} samples", num_batches_test, batchsize, total_samples);
    
    let mut tot_correct = 0i32;
    let fd = openFile(test_path);
    let mut line_buf = vec![0i8; MAX_LINE_SIZE];

    let _ = read_line_utf8(fd, &mut line_buf);
    
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    iter.hash(&mut hasher);
    let hash_val = hasher.finish();
    
    let max_skip = 6000 - (num_batches_test * batchsize);
    let skip_lines = if max_skip > 0 { (hash_val as usize) % max_skip } else { 0 };
    
    for _ in 0..skip_lines {
        if read_line_utf8(fd, &mut line_buf).is_none() {
            break;
        }
    }

    let mut num_actual_batch_tests = 0usize;

    for _i in 0..num_batches_test {
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

        let num_labels: usize = labelsize.iter().sum();

        if num_labels != 0 {
            num_actual_batch_tests += 1;

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
        }
    }

    let total_samples_tested = num_actual_batch_tests * batchsize;
    let final_accuracy = (tot_correct as f64) / (total_samples_tested as f64) * 100.0;
    
    println!("FINAL ACCURACY: {} correct out of {} samples = {:.2}%", tot_correct, total_samples_tested, final_accuracy);
    println!("{} {} {:.6}", iter, 0, final_accuracy / 100.0);
    
    closeFile(fd);
}

fn read_data_svm(num_batches: usize, net: &mut Network, epoch: usize, cfg: &ParsedCfg) {
    let fd = openFile(&cfg.trainData);
    if fd == 0 {
        panic!("Failed to open training data file: {}", cfg.trainData);
    }
    let mut line_buf = vec![0i8; MAX_LINE_SIZE];
    
    if let Some(_header) = read_line_utf8(fd, &mut line_buf) {
        // Skip header line
    }
    
    let mut diagnostics_printed = false;

    for i in 0..num_batches {
        if i > 0 && (i % 500 == 0) {
            eval_data_svm(5, net, (epoch * num_batches + i) as i32, &cfg.testData, cfg.Batchsize);
        }
        
        let mut records: Vec<Vec<usize>> = Vec::with_capacity(cfg.Batchsize);
        let mut values:  Vec<Vec<f32>>   = Vec::with_capacity(cfg.Batchsize);
        let mut sizes:   Vec<usize>      = Vec::with_capacity(cfg.Batchsize);
        let mut labels:  Vec<Vec<usize>> = Vec::with_capacity(cfg.Batchsize);
        let mut labelsize: Vec<usize>    = Vec::with_capacity(cfg.Batchsize);
        
        let mut count = 0usize;
        while count < cfg.Batchsize {
            if let Some(line) = read_line_utf8(fd, &mut line_buf) {
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
            } else {
                // End of file reached - skip this incomplete batch if we don't have enough samples
                if count < cfg.Batchsize {
                    // We don't have a full batch, skip processing this batch entirely
                    break;
                }
            }
        }
        
        if count < cfg.Batchsize {
            continue;
        }
        
        if i == 0 && !diagnostics_printed {
            diagnostics_printed = true;
            println!("First batch: {} samples", count);
        }
        
        let mut rehash = false;
        let mut rebuild = false;

        let batch_index = epoch * num_batches + i;
        
        if cfg.Rehash > 0 && cfg.Batchsize > 0 {
            let rehash_interval = cfg.Rehash / cfg.Batchsize;
            if batch_index % rehash_interval == (rehash_interval - 1) {
                if Mode == 1 || Mode == 4 {
                    rehash = true;
                }
            }
        }
        
        if cfg.Rebuild > 0 && cfg.Batchsize > 0 && cfg.Rehash > 0 {
            let rebuild_interval = cfg.Rebuild / cfg.Batchsize;
            let rehash_interval = cfg.Rehash / cfg.Batchsize;
            if batch_index % rebuild_interval == (rehash_interval - 1) {
                if Mode == 1 || Mode == 4 {
                    rebuild = true;
                }
            }
        }

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
    }
    
    closeFile(fd);
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
    
    let worker_id = if args.len() > 0 && !args[0].contains("slide-in-rust") {
        parse_number(&args[0]) - 1
    } else {
        0
    };

    let mut cfg = ParsedCfg::default();
    
    let config_paths = ["Config_eurlex.csv", "src/Config_eurlex.csv"];
    let mut config_content = String::new();
    let mut config_found = false;
    
    for path in &config_paths {
        if let Ok(content) = std::fs::read_to_string(path) {
            config_content = content;
            config_found = true;
            break;
        }
    }
    
    if !config_found {
        panic!("Failed to read EURLex config file. Tried paths: {:?}", config_paths);
    }
    
    parseconfig(&config_content, &mut cfg);

    cfg.trainData = "dataset\\EURLex-4.3K\\train.txt".to_string();
    cfg.testData = "dataset\\EURLex-4.3K\\test.txt".to_string();

    let num_batches = cfg.totRecords / cfg.Batchsize;
    let num_batches_test = cfg.totRecordsTest / cfg.Batchsize;

    let mut layers_types = vec![NodeType::ReLU; cfg.numLayer.max(1)];
    if !layers_types.is_empty() {
        layers_types[cfg.numLayer - 1] = NodeType::Softmax;
    }

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

    for e in 0..cfg.Epoch {
        println!("EPOCH {}/{}", e + 1, cfg.Epoch);
        
        read_data_svm(num_batches, &mut net, e, &cfg);

        if e == cfg.Epoch - 1 {
            eval_data_svm(num_batches_test, &mut net, ((e + 1) * num_batches) as i32, &cfg.testData, cfg.Batchsize);
        } else {
            eval_data_svm(50, &mut net, ((e + 1) * num_batches) as i32, &cfg.testData, cfg.Batchsize);
        }
        
        net.save_weights(&cfg.savedWeights);
    }

}
