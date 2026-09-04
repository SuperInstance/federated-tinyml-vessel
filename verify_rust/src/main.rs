use std::env;
use std::fs;

const FNV_OFFSET: u64 = 0xCBF29CE484222325;
const FNV_PRIME: u64 = 0x00000100000001B3;

fn fnv1a_64_bytes(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <head.bin>", args[0]);
        std::process::exit(1);
    }
    let bytes = fs::read(&args[1]).expect("read failed");
    let h = fnv1a_64_bytes(&bytes);
    println!("Rust state hash: 0x{:016x}", h);
}
