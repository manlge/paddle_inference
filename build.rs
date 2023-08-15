use std::vec;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=PADDLE_INFERENCE");
    // let paddle_inference_dir = "/home/mark/paddle_inference/paddle_inference_c";
    let paddle_inference_dir =
        std::env::var("PADDLE_INFERENCE").expect("environment PADDLE_INFERENCE");
    let pd_lib_dirs = vec![
        "paddle/lib",
        "third_party/install/onnxruntime/lib",
        "third_party/install/paddle2onnx/lib",
        "third_party/install/mkldnn/lib",
        "third_party/install/mklml/lib",
    ];
    for lib_dir in pd_lib_dirs.iter() {
        println!(
            "cargo:rustc-link-search={}/{}",
            paddle_inference_dir, lib_dir
        );
    }

    // println!("cargo:rustc-link-search=/usr/local/TensorRT-8.4.3.1/lib");
    for lib in [
        "paddle_inference_c",
        "onnxruntime",
        "paddle2onnx",
        "dnnl",
        "iomp5",
    ] {
        println!("cargo:rustc-link-lib=dylib={}", lib);
    }

    let ld_library_path = pd_lib_dirs
        .iter()
        .map(|&s| format!("{paddle_inference_dir}/{s}"))
        .reduce(|acc, item| format!("{acc}:{item}"))
        .unwrap();
    println!(
        "cargo:rustc-env=LD_LIBRARY_PATH={}:{ld_library_path}",
        std::env::var("LD_LIBRARY_PATH").unwrap_or_default()
    );
}
