use std::{env, path::*};
use std::process::Command;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    // TODO: make this crossplatform?
    Command::new("wget")
            .arg("https://anaconda.org/intel/mkl-static/2020.0/download/linux-64/mkl-static-2020.0-intel_166.tar.bz2")
            .args(&["-P", &out_dir]) 
            .status().unwrap();

    Command::new("tar")
            .arg("-xvf")
            .arg(&format!("{}/mkl-static-2020.0-intel_166.tar.bz2", out_dir))
            .args(&["-C", &out_dir])
            .status().unwrap();

    println!("cargo:rustc-link-search={}/lib", out_dir);
    println!("cargo:rustc-link-lib=static-nobundle=mkl_intel_ilp64");
    println!("cargo:rustc-link-lib=static-nobundle=mkl_sequential");
    println!("cargo:rustc-link-lib=static-nobundle=mkl_core");
}
