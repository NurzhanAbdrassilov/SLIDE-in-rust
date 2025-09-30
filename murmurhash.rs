pub fn murmur_hash(key: &[u8], len: u32, seed: u32) -> u32 {
    let c1: u32 = 0xcc9e2d51;
    let c2: u32 = 0x1b873593;
    let r1: u32 = 15;
    let r2: u32 = 13;
    let m: u32 = 5;
    let n: u32 = 0xe6546b64;

    let mut h: u32 = seed;
    let l = (len / 4) as usize;

    for i in 0..l {
        let base = i * 4;
        let chunk = &key[base..base + 4];
        let mut k = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);

        k = k.wrapping_mul(c1);
        k = k.rotate_left(r1);
        k = k.wrapping_mul(c2);

        h ^= k;
        h = h.rotate_left(r2);
        h = h.wrapping_mul(m).wrapping_add(n);
    }

    let tail = &key[(l * 4)..];
    let mut k: u32 = 0;

    match len & 3 {
        3 => {
            k ^= (tail[2] as u32) << 16;
            k ^= (tail[1] as u32) << 8;
            k ^= tail[0] as u32;
            k = k.wrapping_mul(c1);
            k = k.rotate_left(r1);
            k = k.wrapping_mul(c2);
            h ^= k;
        },
        2 => {
            k ^= (tail[1] as u32) << 8;
            k ^= tail[0] as u32;
            k = k.wrapping_mul(c1);
            k = k.rotate_left(r1);
            k = k.wrapping_mul(c2);
            h ^= k;
        },
        1 => {
            k ^= tail[0] as u32;
            k = k.wrapping_mul(c1);
            k = k.rotate_left(r1);
            k = k.wrapping_mul(c2);
            h ^= k;
        },
        _ => {}
    }

    h ^= len;

    h ^= h >> 16;
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2ae35);
    h ^= h >> 16;

    h
}
