use std::collections::HashMap;
use std::cell::{Ref, RefCell, RefMut};
use std::any::TypeId;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write, Seek, SeekFrom, BufReader};
use std::rc::Rc;
use std::mem;
use std::string::String;
use std::path::Path;
use crc32fast::Hasher;
use std::vec::Vec;
use num_complex::{Complex32, Complex64};
use flate2::read::ZlibDecoder;
use zip::read::ZipArchive;
use zip::write::FileOptions;

#[derive(Debug)]
pub struct NpyArray {
    pub shape: Vec<usize>,
    pub word_size: usize,
    pub fortran_order: bool,
    pub num_vals: usize,
    pub data_holder: Rc<RefCell<Vec<u8>>>,
}

impl NpyArray {
    pub fn new(shape: Vec<usize>, word_size: usize, fortran_order: bool) -> Self {
        let num_vals = shape.iter().product();
        let data_holder = Rc::new(RefCell::new(vec![0u8; num_vals * word_size]));
        NpyArray { shape, word_size, fortran_order, num_vals, data_holder }
    }
    pub fn empty() -> Self {
        NpyArray {
            shape: Vec::new(),
            word_size: 0,
            fortran_order: false,
            num_vals: 0,
            data_holder: Rc::new(RefCell::new(Vec::new())),
        }
    }
    pub fn data<T: 'static>(&self) -> Ref<[T]> {
        assert!(self.word_size == std::mem::size_of::<T>(), "dtype size mismatch");
        let len = self.num_vals;
        Ref::map(self.data_holder.borrow(), move |bytes| {
            let ptr = bytes.as_ptr() as *const T;
            unsafe { std::slice::from_raw_parts(ptr, len) }
        })
    }
    pub fn data_mut<T: 'static>(&self) -> RefMut<[T]> {
        assert!(self.word_size == std::mem::size_of::<T>(), "dtype size mismatch");
        let len = self.num_vals;
        RefMut::map(self.data_holder.borrow_mut(), move |bytes| {
            let ptr = bytes.as_mut_ptr() as *mut T;
            unsafe { std::slice::from_raw_parts_mut(ptr, len) }
        })
    }
    pub fn as_vec<T: 'static + Copy>(&self) -> Vec<T> {
        self.data::<T>().to_vec()
    }
    pub fn num_bytes(&self) -> usize {
        self.data_holder.borrow().len()
    }
}

pub type Npz = HashMap<String, NpyArray>;

pub fn npy_save<T: 'static + Copy>(
    fname: &str,
    data: &[T],
    shape: &[usize],
    mode: &str,
) -> io::Result<()> {
    let file_exists = Path::new(fname).exists();
    let mut f = if mode == "a" && file_exists {
        OpenOptions::new().read(true).write(true).open(fname)?
    } else {
        File::create(fname)?
    };

    let true_shape = if mode == "a" && file_exists {
        //file exists. we need to append to it. read the header, modify the array size
        let mut old_shape = Vec::new();
        let mut word_size = 0usize;
        let mut fortran_order = false;
        parse_npy_header_from_file(&mut f, &mut word_size, &mut old_shape, &mut fortran_order)?;
        assert!(!fortran_order, "fortran_order must be False");
        assert_eq!(word_size, mem::size_of::<T>(), "type mismatch on append");
        assert_eq!(old_shape.len(), shape.len(), "dimensionality mismatch on append");
        for i in 1..shape.len() {
            assert_eq!(old_shape[i], shape[i], "shape[{}] mismatch", i);
        }
        old_shape[0] += shape[0];
        old_shape
    } else {
        shape.to_vec()
    };

    let header = create_npy_header::<T>(&true_shape);
    f.seek(SeekFrom::Start(0))?;
    f.write_all(&header)?;

    f.seek(SeekFrom::End(0))?;
    let nels = shape.iter().product::<usize>();
    let data_bytes = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            nels * mem::size_of::<T>(),
        )
    };
    f.write_all(data_bytes)?;

    Ok(())
}

pub fn npz_save<T: Copy + 'static>(
    zipname: &str,
    fname: &str,
    data: &[T],
    shape: &[usize],
    mode: &str,
) -> io::Result<()> {
    let entry_name = format!("{}.npy", fname);

    let mut nrecs: u16 = 0;
    let mut global_header_size: usize = 0;
    let mut global_header_offset: usize = 0;
    let mut global_header: Vec<u8> = Vec::new();

    let mut f = if mode == "a" {
        //zip file exists. we need to add a new npy file to it.
            //first read the footer. this gives us the offset and size of the global header
            //then read and store the global header.
            //below, we will write the the new data at the start of the global header then append the global header and footer below it
        let mut f = OpenOptions::new().read(true).write(true).open(zipname)?;
        parse_zip_footer(&mut f, &mut nrecs, &mut global_header_size, &mut global_header_offset)?;
        f.seek(SeekFrom::Start(global_header_offset as u64))?;
        global_header.resize(global_header_size, 0);
        f.read_exact(&mut global_header)?;
        f.seek(SeekFrom::Start(global_header_offset as u64))?;
        f
    } else {
        File::create(zipname)?
    };

    let npy_header = create_npy_header::<T>(shape);
    let nels = shape.iter().product::<usize>();
    let nbytes = npy_header.len() + nels * mem::size_of::<T>();

    let mut hasher = Hasher::new();
    hasher.update(&npy_header);
    let data_bytes = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            nels * mem::size_of::<T>(),
        )
    };
    //get the CRC of the data to be added
    hasher.update(data_bytes);
    let crc = hasher.finalize();

    //build the local header
    let mut local_header = Vec::new();
    local_header.extend(b"PK"); //first part of sig
    local_header.extend(&0x0403u16.to_le_bytes()); //second part of sig
    local_header.extend(&20u16.to_le_bytes()); //min version to extract
    local_header.extend(&0u16.to_le_bytes()); // general purpose bit flag
    local_header.extend(&0u16.to_le_bytes()); // compression method
    local_header.extend(&0u16.to_le_bytes()); //file last mod time
    local_header.extend(&0u16.to_le_bytes()); //file last mod date
    local_header.extend(&crc.to_le_bytes()); // crc
    local_header.extend(&(nbytes as u32).to_le_bytes()); // compressed size
    local_header.extend(&(nbytes as u32).to_le_bytes()); // uncompressed size
    local_header.extend(&(entry_name.len() as u16).to_le_bytes()); // filename length
    local_header.extend(&0u16.to_le_bytes());         // extra field length
    local_header.extend(entry_name.as_bytes());       // filename

    //build global header
    {
        let mut cen: Vec<u8> = Vec::new();
        cen.extend(b"PK"); //first part of sig
        cen.extend(&0x0201u16.to_le_bytes()); //second part of sig
        cen.extend(&20u16.to_le_bytes()); //version made by
        cen.extend(&local_header[4..30]);
        cen.extend(&0u16.to_le_bytes()); //file comment length
        cen.extend(&0u16.to_le_bytes()); //disk number where file starts
        cen.extend(&0u16.to_le_bytes()); //internal file attributes
        cen.extend(&0u32.to_le_bytes()); //external file attributes
        //relative offset of local file header, since it begins where the global header used to begin
        cen.extend(&(global_header_offset as u32).to_le_bytes());
        cen.extend(entry_name.as_bytes());
        global_header.extend(cen);
    }

    //build footer
    let mut footer = Vec::new();
    footer.extend(b"PK"); //first part of sig
    footer.extend(&0x0605u16.to_le_bytes()); //second part of sig
    footer.extend(&0u16.to_le_bytes()); //number of this disk
    footer.extend(&0u16.to_le_bytes()); //disk where footer starts
    footer.extend(&(nrecs.wrapping_add(1)).to_le_bytes()); //number of records on this disk
    footer.extend(&(nrecs.wrapping_add(1)).to_le_bytes()); //total number of records
    footer.extend(&(global_header.len() as u32).to_le_bytes()); //nbytes of global headers
    //offset of start of global headers, since global header now starts after newly written array
    let eocd_offset = (global_header_offset as u32)
        .wrapping_add((local_header.len() + nbytes) as u32);
    footer.extend(&eocd_offset.to_le_bytes());
    footer.extend(&0u16.to_le_bytes());//zip file comment length

    //write everything
    f.write_all(&local_header)?;
    f.write_all(&npy_header)?;
    f.write_all(data_bytes)?;
    f.write_all(&global_header)?;
    f.write_all(&footer)?;
    f.flush()?;

    Ok(())
}

pub fn npy_save_vec<T: 'static + Copy>(fname: &str, data: &Vec<T>, mode: &str) -> std::io::Result<()> {
    let shape = vec![data.len()];
    npy_save(fname, data, &shape, mode)
}

pub fn npz_save_vec<T: 'static + Copy>(zipname: &str, fname: &str, data: &Vec<T>, mode: &str) -> std::io::Result<()> {
    let shape = vec![data.len()];
    npz_save(zipname, fname, data, &shape, mode)
}


pub fn create_npy_header<T: Copy + 'static>(shape: &[usize]) -> Vec<u8> {
    let mut dict = String::new();
    dict += "{'descr': '";
    dict.push(big_endian_test());
    dict.push(map_type::<T>());
    dict += &format!("{}', ", std::mem::size_of::<T>());
    dict += "'fortran_order': False, 'shape': (";
    dict += &shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ");
    if shape.len() == 1 { dict += ","; }
    dict += "), }";
    //pad with spaces so that preamble+dict is modulo 16 bytes. preamble is 10 bytes. dict needs to end with \n
    let preamble_len = 10;
    let mut header_len = dict.len() + 1; // +1 for newline
    let pad = (16 - ((preamble_len + header_len) % 16)) % 16;
    dict.extend(std::iter::repeat(' ').take(pad));
    dict.push('\n'); 
    header_len = dict.len();
    println!("HEADER STRING: [{}]", dict);
    println!("HEADER LEN: {}", header_len);
    let mut header = vec![0x93u8];
    header.extend(b"NUMPY");
    header.push(0x01); // //major version of numpy format
    header.push(0x00); // minor version of numpy format
    let dict_len = header_len as u16;
    header.extend(&dict_len.to_le_bytes());
    header.extend(dict.as_bytes());
    println!("HEADER BYTES: {:?}", &header);
    header
}

pub fn big_endian_test() -> char {
    let x: u32 = 1;
    let x_bytes = x.to_le_bytes();
    if x_bytes[0] == 1 { '<' } else { '>' }
}

pub fn map_type<T: 'static>() -> char {
    let t = TypeId::of::<T>();
    if t == TypeId::of::<f32>() { 'f' }
    else if t == TypeId::of::<f64>() { 'f' }
    else if t == TypeId::of::<i8>() { 'i' }
    else if t == TypeId::of::<i16>() { 'i' }
    else if t == TypeId::of::<i32>() { 'i' }
    else if t == TypeId::of::<i64>() { 'i' }
    else if t == TypeId::of::<u8>() { 'u' }
    else if t == TypeId::of::<u16>() { 'u' }
    else if t == TypeId::of::<u32>() { 'u' }
    else if t == TypeId::of::<u64>() { 'u' }
    else if t == TypeId::of::<bool>() { 'b' }
    else if t == TypeId::of::<Complex32>() { 'c' }
    else if t == TypeId::of::<Complex64>() { 'c' }
    else { '?' }
}

pub fn parse_npy_header_from_buffer(
    buffer: &[u8],
    word_size: &mut usize,
    shape: &mut Vec<usize>,
    fortran_order: &mut bool,
) {
    let major_version = buffer[6];
    let minor_version = buffer[7];
    let header_len = u16::from_le_bytes([buffer[8], buffer[9]]) as usize;
    let header = String::from_utf8_lossy(&buffer[10..10 + header_len]);

    // fortran order
    let loc1 = header.find("fortran_order").unwrap() + 16;
    *fortran_order = &header[loc1..loc1 + 4] == "True";

    // shape
    let loc1 = header.find('(').unwrap();
    let loc2 = header.find(')').unwrap();
    let mut str_shape = header[loc1 + 1..loc2].to_string();
    shape.clear();
    for s in str_shape.split(',') {
        let s = s.trim();
        if !s.is_empty() {
            if let Ok(val) = s.parse::<usize>() {
                shape.push(val);
            }
        }
    }

    // endian, word size, data type
    let loc1 = header.find("descr").unwrap() + 9;
    let little_endian = {
        let c = header.chars().nth(loc1).unwrap();
        c == '<' || c == '|'
    };
    assert!(little_endian);

    let str_ws = &header[loc1 + 2..];
    let loc2 = str_ws.find('\'').unwrap();
    *word_size = str_ws[..loc2].parse::<usize>().unwrap();
}

pub fn parse_npy_header_from_file<R: Read + Seek>(
    fp: &mut R,
    word_size: &mut usize,
    shape: &mut Vec<usize>,
    fortran_order: &mut bool,
) -> std::io::Result<()> {
    let mut preamble = [0u8; 10];
    fp.read_exact(&mut preamble)?;
    let header_len = u16::from_le_bytes([preamble[8], preamble[9]]) as usize;
    let mut header = vec![0u8; header_len];
    fp.read_exact(&mut header)?;
    let header_str = String::from_utf8_lossy(&header);
    if !header_str.ends_with('\n') {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "parse_npy_header: header does not end with newline"));
    }
    // fortran order
    let loc1 = header_str.find("fortran_order").ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "parse_npy_header: failed to find header keyword: 'fortran_order'"))? + 16;
    *fortran_order = &header_str[loc1..loc1 + 4] == "True";
    // shape
    let loc1 = header_str.find('(').ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "parse_npy_header: failed to find header keyword: '('"))?;
    let loc2 = header_str.find(')').ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "parse_npy_header: failed to find header keyword: ')'"))?;
    let str_shape = header_str[loc1 + 1..loc2].to_string();
    shape.clear();
    for s in str_shape.split(',') {
        let s = s.trim();
        if !s.is_empty() {
            if let Ok(val) = s.parse::<usize>() {
                shape.push(val);
            }
        }
    }
    // endian, word size, data type
    let loc1 = header_str.find("descr").ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "parse_npy_header: failed to find header keyword: 'descr'"))? + 9;
    let little_endian = {
        let c = header_str.chars().nth(loc1).unwrap();
        c == '<' || c == '|'
    };
    assert!(little_endian);
    let str_ws = &header_str[loc1 + 2..];
    let loc2 = str_ws.find('\'').unwrap();
    *word_size = str_ws[..loc2].parse::<usize>().unwrap();
    Ok(())
}

pub fn parse_zip_footer<R: Read + Seek>(fp: &mut R, nrecs: &mut u16, global_header_size: &mut usize, global_header_offset: &mut usize) -> std::io::Result<()> {
    let mut footer = vec![0u8; 22];
    fp.seek(SeekFrom::End(-22))?;
    fp.read_exact(&mut footer)?;
    let disk_no = u16::from_le_bytes([footer[4], footer[5]]);
    let disk_start = u16::from_le_bytes([footer[6], footer[7]]);
    let nrecs_on_disk = u16::from_le_bytes([footer[8], footer[9]]);
    *nrecs = u16::from_le_bytes([footer[10], footer[11]]);
    *global_header_size = u32::from_le_bytes([footer[12], footer[13], footer[14], footer[15]]) as usize;
    *global_header_offset = u32::from_le_bytes([footer[16], footer[17], footer[18], footer[19]]) as usize;
    let comment_len = u16::from_le_bytes([footer[20], footer[21]]);
    assert_eq!(disk_no, 0);
    assert_eq!(disk_start, 0);
    assert_eq!(nrecs_on_disk, *nrecs);
    assert_eq!(comment_len, 0);
    Ok(())
}

fn load_the_npy_file<R: Read + Seek>(fp: &mut R) -> std::io::Result<NpyArray> {
    let mut shape = Vec::new();
    let mut word_size = 0;
    let mut fortran_order = false;
    parse_npy_header_from_file(fp, &mut word_size, &mut shape, &mut fortran_order)?;
    let num_vals: usize = shape.iter().product();
    let mut arr = NpyArray::new(shape, word_size, fortran_order);
    let mut data = vec![0u8; arr.num_bytes()];
    fp.read_exact(&mut data)?;
    *arr.data_holder.borrow_mut() = data;
    Ok(arr)
}

pub fn load_the_npz_array<R: Read + Seek>(fp: &mut R, compr_bytes: u32, uncompr_bytes: u32) -> std::io::Result<NpyArray> {
    let mut buffer_compr = vec![0u8; compr_bytes as usize];
    fp.read_exact(&mut buffer_compr)?;

    let mut decoder = ZlibDecoder::new(&buffer_compr[..]);
    let mut buffer_uncompr = vec![0u8; uncompr_bytes as usize];
    decoder.read_exact(&mut buffer_uncompr)?;

    let mut shape = Vec::new();
    let mut word_size = 0;
    let mut fortran_order = false;
    parse_npy_header_from_buffer(&buffer_uncompr, &mut word_size, &mut shape, &mut fortran_order);

    let mut array = NpyArray::new(shape, word_size, fortran_order);

    let offset = uncompr_bytes as usize - array.num_bytes();
    let data = &buffer_uncompr[offset..offset + array.num_bytes()];
    *array.data_holder.borrow_mut() = data.to_vec();

    Ok(array)
}

pub fn npz_load(fname: &str) -> std::io::Result<Npz> {
    let file = File::open(fname)?;
    let mut archive = ZipArchive::new(file)?;
    let mut arrays = Npz::new();

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let name = file.name().to_string();
        if !name.ends_with(".npy") {
            continue;
        }
        let varname = name.trim_end_matches(".npy").to_string();

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let mut shape = Vec::new();
        let mut word_size = 0;
        let mut fortran_order = false;
        parse_npy_header_from_buffer(&buffer, &mut word_size, &mut shape, &mut fortran_order);
        let mut arr = NpyArray::new(shape, word_size, fortran_order);
        let offset = buffer.len() - arr.num_bytes();
        let data = &buffer[offset..];
        *arr.data_holder.borrow_mut() = data.to_vec();
        arrays.insert(varname, arr);
    }
    Ok(arrays)
}

pub fn npz_load_var(fname: &str, varname: &str) -> std::io::Result<NpyArray> {
    let file = File::open(fname)?;
    let mut archive = ZipArchive::new(file)?;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let name = file.name().to_string();
        if !name.ends_with(".npy") {
            continue;
        }
        let vname = name.trim_end_matches(".npy");
        if vname == varname {
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;

            let mut shape = Vec::new();
            let mut word_size = 0;
            let mut fortran_order = false;
            parse_npy_header_from_buffer(&buffer, &mut word_size, &mut shape, &mut fortran_order);
            let mut arr = NpyArray::new(shape, word_size, fortran_order);
            let offset = buffer.len() - arr.num_bytes();
            let data = &buffer[offset..];
            *arr.data_holder.borrow_mut() = data.to_vec();
            return Ok(arr);
        }
    }
    Err(std::io::Error::new(std::io::ErrorKind::NotFound, "Variable not found in npz"))
}

pub fn npy_load(fname: &str) -> std::io::Result<NpyArray> {
    let mut file = File::open(fname)?;
    load_the_npy_file(&mut file)
}