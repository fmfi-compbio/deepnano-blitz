#![feature(static_nobundle)]

#[macro_use(s)]

extern crate ndarray;
extern crate kth;
extern crate libc;

mod matrix_load;
mod conv_layer;
mod gru_layer;
mod approx;
mod beam_search;
mod models;

use beam_search::*;
use models::*;
use ndarray::{concatenate, Axis};

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::{PyArray2};
use libc::{c_float, c_int, c_longlong};

const STEP: usize = 500;
const PAD: usize = 10;

extern "C" {
    fn vmsExp(n: c_int, a: *const c_float, y: *mut c_float, mode: c_longlong);
}

#[pyclass(unsendable)]
struct Caller {
    net: Box<dyn Net>,
    beam_size: usize,
    beam_cut_threshold: f32
}

#[pymethods]
impl Caller {
    #[new]
    fn new(net_type: &str, path: &str, beam_size: usize, beam_cut_threshold: f32) -> Self {
        let net: Box<dyn Net> = if net_type == "256" {
            Box::new(NetBig::new(&path).unwrap())
        } else if net_type == "56" { 
            Box::new(Net56::new(&path).unwrap())
        } else if net_type == "64" { 
            Box::new(Net64::new(&path).unwrap())
        } else if net_type == "80" { 
            Box::new(Net80::new(&path).unwrap())
        } else if net_type == "96" { 
            Box::new(Net96::new(&path).unwrap())
        } else {
            Box::new(NetSmall::new(&path).unwrap())
        };
        Caller {
            net, beam_size, beam_cut_threshold
        }
    }
    
    fn call_raw_signal(&mut self, raw_data: Vec<f32>) -> (String,String) {
        let mut start_pos = 0;

        let mut to_stack = Vec::new();

        while start_pos + STEP * 3 + PAD * 6 < raw_data.len() {
            let chunk = &raw_data[start_pos..(start_pos + STEP * 3 + PAD * 6).min(raw_data.len())];
            let out = self.net.predict(chunk);

            let slice_start_pos = if start_pos == 0 {
                0
            } else {
                (PAD).min(out.shape()[0])
            };
            let slice_end_pos = if start_pos + 3 * STEP + 6 * PAD >= raw_data.len() {
                out.shape()[0]
            } else {
                out.shape()[0] - PAD
            };

            to_stack.push(out.slice(s![slice_start_pos..slice_end_pos, ..]).to_owned());

            start_pos += 3 * STEP;
        }

        if to_stack.len() == 0 {
            return (String::new(), String::new())
        }

        let mut result = concatenate(
            Axis(0),
            &(to_stack.iter().map(|x| x.view()).collect::<Vec<_>>()),
        ).unwrap();

        /*if self.beam_size > 1*/ {
            unsafe {
                let ptr = result.as_mut_ptr();
                vmsExp(5 * result.shape()[0] as i32, ptr, ptr, 259);
            }
            for mut row in result.outer_iter_mut() {
                let sum = row.sum();
                row.mapv_inplace(|x| {
                    x / sum
                });
            }

            beam_search(&result, self.beam_size, self.beam_cut_threshold)
        } /*else {
            let alphabet: Vec<char> = "NACGT".chars().collect();

            let mut out = String::new();
            let mut out_p = String::new();
            let preds = result
                .outer_iter()
                .map(|sample_predict| {
                    let best = sample_predict.iter().enumerate().fold(0, |best, (i, &x)| {
                        if x > sample_predict[best] {
                            i
                        } else {
                            best
                        }
                    });
                    best
                })
                .scan(0, |state, x| {
                    let ret = (*state, x);
                    *state = x;
                    Some(ret)
                })
                .filter_map(|(prev, current)| {
                    if prev == current || current == 0 {
                        None
                    } else {
                        Some(alphabet[current])
                    }
                })
                .collect::<String>();

            preds 
        }*/
    }
}



#[pyfunction]
fn beam_search_py(result: &PyArray2<f32>, beam_size: usize, beam_cut_threshold: f32) -> String {
    unsafe {
        beam_search(&result.as_array(), beam_size, beam_cut_threshold).0
    }
}

#[pymodule]
fn deepnano2(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Caller>()?;
    m.add_wrapped(wrap_pyfunction!(beam_search_py))?;

    Ok(())
}
