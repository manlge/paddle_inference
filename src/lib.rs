#[cfg(feature = "serde")]
#[cfg_attr(feature = "serde", macro_use)]
extern crate serde;

pub mod common;
pub mod config;
pub mod ctypes {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(unused)]

    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
mod predictor;
mod tensor;
pub mod utils;

pub use predictor::Predictor;
pub use tensor::Tensor;
