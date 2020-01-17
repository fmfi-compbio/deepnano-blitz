#![feature(static_nobundle)]

#[macro_use(s, array)]
extern crate ndarray;
extern crate kth;
extern crate libc;

mod matrix_load;
mod conv_layer;
mod gru_layer;

use matrix_load::*;
use conv_layer::*;
use gru_layer::*;

use libc::c_void;
use ndarray::{stack, Array, Axis, Ix1, Ix2};
use std::cell::RefCell;
use std::error::Error;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::ptr;
use std::env;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

extern "C" {
    fn mkl_cblas_jit_create_sgemm(
        JITTER: *mut *mut c_void,
        layout: u32,
        transa: u32,
        transb: u32,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        lda: usize,
        ldb: usize,
        beta: f32,
        ldc: usize,
    ) -> u32;
    fn mkl_jit_get_sgemm_ptr(JITTER: *const c_void) -> SgemmJitKernelT;
}


const STEP: usize = 1000;
const PAD: usize = 10;

struct OutLayer {
    w: Array<f32, Ix2>,
    b: Array<f32, Ix1>,
}

impl OutLayer {
    fn new<R: BufRead>(f: &mut R) -> Result<OutLayer, Box<dyn Error>> {
        Ok(OutLayer {
            w: load2dmatrix(f)?,
            b: load1dmatrix(f)?,
        })
    }

    fn calc(&self, input: &Array<f32, Ix2>) -> Array<f32, Ix2> {
        let logits = input.dot(&self.w) + &self.b;
        logits
    }
}

struct Conv33_48 {
}

impl ConvSizer for Conv33_48 {
  fn input_features() -> usize { 2 }
  fn output_features() -> usize { 48 }
  fn conv_filter_size() -> usize { 33 }
  fn sequence_size() -> usize { 3*STEP + 6*PAD }
  fn pool_kernel() -> usize { 3 }
}

struct GRU48 {
}

static mut JITTER48: *mut c_void = ptr::null_mut();
static mut SGEMM48: SgemmJitKernelT = None;

impl GRUSizer for GRU48 {
  fn sequence_size() -> usize { STEP + 2*PAD }
  fn output_features() -> usize { 48 }
  fn jitter() -> *mut c_void { unsafe { JITTER48 } }
  fn sgemm() -> SgemmJitKernelT { unsafe { SGEMM48 } }
}

struct Net {
    conv_layer1: ConvLayer<Conv33_48>,
    rnn_layer1: BiGRULayer<GRU48>,
    rnn_layer2: BiGRULayer<GRU48>,
    out_layer: OutLayer,
}

impl Net {
    fn new(filename: &str) -> Result<Net, Box<dyn Error>> {
        let netfile = File::open(filename)?;
        let mut bufnetfile = BufReader::new(&netfile);
        let conv_layer1 = ConvLayer::new(&mut bufnetfile)?;
        let rnn_layer1 = BiGRULayer::new(&mut bufnetfile)?;
        let rnn_layer2 = BiGRULayer::new(&mut bufnetfile)?;
        let out_layer = OutLayer::new(&mut bufnetfile)?;
        Ok(Net {
            conv_layer1,
            rnn_layer1,
            rnn_layer2,
            out_layer,
        })
    }

    fn predict(&mut self, chunk: &[f32]) -> Array<f32, Ix2> {
        let scaled_data = Array::from_shape_vec(
            (chunk.len(), 2),
            chunk
                .iter()
                .flat_map(|&x| {
                    use std::iter::once;
                    let scaled = x.max(-2.5).min(2.5);
                    once(scaled).chain(once(scaled * scaled))
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let r1 = self.conv_layer1.calc(&scaled_data);
        let r2 = self.rnn_layer1.calc(&r1);
        let r3 = self.rnn_layer2.calc(&r2);
        let out = self.out_layer.calc(&r3);

        out
    }
}

#[pyclass]
struct Caller {
    net: Net
}

fn lg_sum(a: f32, b: f32) -> f32 {
    if a == std::f32::NEG_INFINITY {
        b
    } else if b == std::f32::NEG_INFINITY {
        a
    } else if a > b {
        (b-a).exp().ln_1p() + a
    } else {
        (a-b).exp().ln_1p() + b
    }
}

#[pymethods]
impl Caller {
    #[new]
    fn new(obj: &PyRawObject, path: &str) {
        obj.init({
            Caller {
                net: Net::new(&path).unwrap()
            }
        })
    }
    
    fn call_raw_signal(&mut self, raw_data: Vec<f32>) -> String {
        unsafe {
          if JITTER48 == ptr::null_mut() {
            initialize_jit48();
          }
        }
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
            return String::new()
        }

        let result = stack(
            Axis(0),
            &(to_stack.iter().map(|x| x.view()).collect::<Vec<_>>()),
        ).unwrap();
        let alphabet: Vec<char> = "NACGT".chars().collect();

/*        let preds = result
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
            .collect::<String>();*/


        let out = bs(&result);
/*        println!("{} {}", out.len(), preds.len());*/

/*        println!("{:?}", bs(
            &array![[1., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0.],
                    [1., 0., 0., 0., 0.]]));
        println!("{:?}", bs(
            &array![[1., 0., 0., 0., 0.],
                    [1., 0.9, 0., 0., 0.],
                    [1., 0.9, 0., 0., 0.],
                    [1., 0., 0., 0., 0.]]));*/
        out
    }
}

fn bs(result: &Array<f32, Ix2>) -> String {
    let alphabet: Vec<char> = "NACGT".chars().collect();
    let beam_size = 5;
    // (base, what)
    let mut beam_prevs = vec![(0, 0)];
    let mut beam_forward: Vec<[i32; 4]> = vec![[-1, -1, -1, -1]];
    let mut cur_probs = vec![(0i32, std::f32::NEG_INFINITY, 0.0)];
    let mut new_probs = Vec::new();
    
    for pr in result.slice(s![..;-1, ..]).outer_iter() {
        new_probs.clear();

        for &(beam, base_prob, n_prob) in &cur_probs {
            // add N to beam
            new_probs.push((beam, std::f32::NEG_INFINITY, lg_sum(n_prob, base_prob) + pr[0]));

            for b in 1..5 {
                if b == beam_prevs[beam as usize].0 {
                    new_probs.push((beam, base_prob + pr[b], std::f32::NEG_INFINITY));
                    let mut new_beam = beam_forward[beam as usize][b-1];
                    if new_beam == -1 {
                        new_beam = beam_prevs.len() as i32;
                        beam_prevs.push((b, beam));
                        beam_forward[beam as usize][b-1] = new_beam;
                        beam_forward.push([-1, -1, -1, -1]);
                    }

                    new_probs.push((new_beam, n_prob + pr[b], std::f32::NEG_INFINITY));

                } else {
                    let mut new_beam = beam_forward[beam as usize][b-1];
                    if new_beam == -1 {
                        new_beam = beam_prevs.len() as i32;
                        beam_prevs.push((b, beam));
                        beam_forward[beam as usize][b-1] = new_beam;
                        beam_forward.push([-1, -1, -1, -1]);
                    }

                    new_probs.push((new_beam, lg_sum(base_prob, n_prob) + pr[b], std::f32::NEG_INFINITY));
                }
            }
        }
        std::mem::swap(&mut cur_probs, &mut new_probs);

        cur_probs.sort_by_key(|x| x.0);
        let mut last_key: i32 = -1;
        let mut last_key_pos = 0;
        for i in 0..cur_probs.len() {
            if cur_probs[i].0 == last_key {
                cur_probs[last_key_pos].1 = lg_sum(cur_probs[last_key_pos].1, cur_probs[i].1);
                cur_probs[last_key_pos].2 = lg_sum(cur_probs[last_key_pos].2, cur_probs[i].2);
                cur_probs[i].0 = -1;
            } else {
                last_key_pos = i;
                last_key = cur_probs[i].0;
            }
        }

        cur_probs.retain(|x| x.0 != -1);
        cur_probs.sort_by(|a, b| lg_sum(b.1, b.2).partial_cmp(&lg_sum(a.1, a.2)).unwrap());
        cur_probs.truncate(beam_size);
    }

    let mut out = String::new();
    let mut beam = cur_probs[0].0;
    while beam != 0 {
        out.push(alphabet[beam_prevs[beam as usize].0]);
        beam = beam_prevs[beam as usize].1;
    }
    out
}

struct Conv33_256 {
}

impl ConvSizer for Conv33_256 {
  fn input_features() -> usize { 2 }
  fn output_features() -> usize { 256 }
  fn conv_filter_size() -> usize { 33 }
  fn sequence_size() -> usize { 3*STEP + 6*PAD }
  fn pool_kernel() -> usize { 3 }
}

struct GRU256 {
}

static mut JITTER256: *mut c_void = ptr::null_mut();
static mut SGEMM256: SgemmJitKernelT = None;

impl GRUSizer for GRU256 {
  fn sequence_size() -> usize { STEP + 2*PAD }
  fn output_features() -> usize { 256 }
  fn jitter() -> *mut c_void { unsafe { JITTER256 } }
  fn sgemm() -> SgemmJitKernelT { unsafe { SGEMM256 } }
}
struct NetBig {
    conv_layer1: ConvLayer<Conv33_256>,
    rnn_layer1: BiGRULayer<GRU256>,
    rnn_layer2: BiGRULayer<GRU256>,
    rnn_layer3: BiGRULayer<GRU256>,
    out_layer: OutLayer,
}

impl NetBig {
    fn new(filename: &str) -> Result<NetBig, Box<dyn Error>> {
        let netfile = File::open(filename)?;
        let mut bufnetfile = BufReader::new(&netfile);
        let conv_layer1 = ConvLayer::new(&mut bufnetfile)?;
        let rnn_layer1 = BiGRULayer::new(&mut bufnetfile)?;
        let rnn_layer2 = BiGRULayer::new(&mut bufnetfile)?;
        let rnn_layer3 = BiGRULayer::new(&mut bufnetfile)?;
        let out_layer = OutLayer::new(&mut bufnetfile)?;
        Ok(NetBig {
            conv_layer1,
            rnn_layer1,
            rnn_layer2,
            rnn_layer3,
            out_layer,
        })
    }

    fn predict(&mut self, chunk: &[f32]) -> Array<f32, Ix2> {
        let scaled_data = Array::from_shape_vec(
            (chunk.len(), 2),
            chunk
                .iter()
                .flat_map(|&x| {
                    use std::iter::once;
                    let scaled = x.max(-2.5).min(2.5);
                    once(scaled).chain(once(scaled * scaled))
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let r1 = self.conv_layer1.calc(&scaled_data);
        let r2 = self.rnn_layer1.calc(&r1);
        let r3 = self.rnn_layer2.calc(&r2);
        let r4 = self.rnn_layer3.calc(&r3);
        let out = self.out_layer.calc(&r4);

        out
    }
}

#[pyclass]
struct CallerBig {
    net: NetBig
}

#[pymethods]
impl CallerBig {
    #[new]
    fn new(obj: &PyRawObject, path: &str) {
        obj.init({
            CallerBig {
                net: NetBig::new(&path).unwrap()
            }
        })
    }
    
    fn call_raw_signal(&mut self, raw_data: Vec<f32>) -> String {
        unsafe {
          if JITTER256 == ptr::null_mut() {
            initialize_jit256();
          }
        }
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
            return String::new()
        }

        let result = stack(
            Axis(0),
            &(to_stack.iter().map(|x| x.view()).collect::<Vec<_>>()),
        ).unwrap();
        let alphabet: Vec<char> = "NACGT".chars().collect();

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
    }
}

fn initialize_jit48() {
    unsafe {
        JITTER48 = ptr::null_mut();
        // TODO: check
        let status = mkl_cblas_jit_create_sgemm(
            &mut JITTER48,
            101,
            111,
            111,
            1,
            48 * 3,
            48,
            1.0,
            48,
            48 * 3,
            0.0,
            48 * 3,
        );

        SGEMM48 = mkl_jit_get_sgemm_ptr(JITTER48);
    }
}

fn initialize_jit256() {
    unsafe {
        JITTER256 = ptr::null_mut();
        // TODO: check
        let status = mkl_cblas_jit_create_sgemm(
            &mut JITTER256,
            101,
            111,
            111,
            1,
            256 * 3,
            256,
            1.0,
            256,
            256 * 3,
            0.0,
            256 * 3,
        );

        SGEMM256 = mkl_jit_get_sgemm_ptr(JITTER256);
    }
}
#[pymodule]
fn deepnano2(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Caller>()?;
    m.add_class::<CallerBig>()?;

    Ok(())
}
