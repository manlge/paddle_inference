use std::{env, path::PathBuf, vec};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=wrapper.h");

    println!("cargo:rerun-if-env-changed=PADDLE_INFERENCE");
    let paddle_inference_dir =
        std::env::var("PADDLE_INFERENCE").expect("environment PADDLE_INFERENCE");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}/paddle/include", paddle_inference_dir))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
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
