#![feature(static_nobundle)]

#[macro_use(s, array)]
extern crate ndarray;
extern crate kth;
extern crate libc;

mod matrix_load;
mod conv_layer;
mod gru_layer;
mod approx;

use matrix_load::*;
use conv_layer::*;
use gru_layer::*;

use libc::{c_float, c_int, c_longlong, c_void};
use ndarray::{stack, Array, Axis, Ix1, Ix2, ArrayBase, Data};
use std::error::Error;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::ptr;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::{PyArray2};

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
    fn vmsExp(n: c_int, a: *const c_float, y: *mut c_float, mode: c_longlong);
}


const STEP: usize = 500;
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
        input.dot(&self.w) + &self.b
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

trait Net {
    fn predict(&mut self, chunk: &[f32]) -> Array<f32, Ix2>;
}

struct NetSmall {
    conv_layer1: ConvLayer<Conv33_48>,
    rnn_layer1: BiGRULayer<GRU48>,
    rnn_layer2: BiGRULayer<GRU48>,
    out_layer: OutLayer,
}

impl NetSmall {
    fn new(filename: &str) -> Result<NetSmall, Box<dyn Error>> {
        let netfile = File::open(filename)?;
        let mut bufnetfile = BufReader::new(&netfile);
        let conv_layer1 = ConvLayer::new(&mut bufnetfile)?;
        let rnn_layer1 = BiGRULayer::new(&mut bufnetfile)?;
        let rnn_layer2 = BiGRULayer::new(&mut bufnetfile)?;
        let out_layer = OutLayer::new(&mut bufnetfile)?;
        Ok(NetSmall {
            conv_layer1,
            rnn_layer1,
            rnn_layer2,
            out_layer,
        })
    }
}

impl Net for NetSmall {
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

        self.conv_layer1.calc(&scaled_data);
        self.rnn_layer1.calc(&self.conv_layer1.pooled_out);
        self.rnn_layer2.calc(&self.rnn_layer1.bwd.output);
        let out = self.out_layer.calc(&self.rnn_layer2.bwd.output);

        out
    }
}

struct Conv33_56 {
}

impl ConvSizer for Conv33_56 {
  fn input_features() -> usize { 2 }
  fn output_features() -> usize { 56 }
  fn conv_filter_size() -> usize { 33 }
  fn sequence_size() -> usize { 3*STEP + 6*PAD }
  fn pool_kernel() -> usize { 3 }
}

struct GRU56 {
}

static mut JITTER56: *mut c_void = ptr::null_mut();
static mut SGEMM56: SgemmJitKernelT = None;

impl GRUSizer for GRU56 {
  fn sequence_size() -> usize { STEP + 2*PAD }
  fn output_features() -> usize { 56 }
  fn jitter() -> *mut c_void { unsafe { JITTER56 } }
  fn sgemm() -> SgemmJitKernelT { unsafe { SGEMM56 } }
}

struct Net56 {
    conv_layer1: ConvLayer<Conv33_56>,
    rnn_layer1: BiGRULayer<GRU56>,
    rnn_layer2: BiGRULayer<GRU56>,
    out_layer: OutLayer,
}

impl Net56 {
    fn new(filename: &str) -> Result<Net56, Box<dyn Error>> {
        let netfile = File::open(filename)?;
        let mut bufnetfile = BufReader::new(&netfile);
        let conv_layer1 = ConvLayer::new(&mut bufnetfile)?;
        let rnn_layer1 = BiGRULayer::new(&mut bufnetfile)?;
        let rnn_layer2 = BiGRULayer::new(&mut bufnetfile)?;
        let out_layer = OutLayer::new(&mut bufnetfile)?;
        Ok(Net56 {
            conv_layer1,
            rnn_layer1,
            rnn_layer2,
            out_layer,
        })
    }
}

impl Net for Net56 {
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

        self.conv_layer1.calc(&scaled_data);
        self.rnn_layer1.calc(&self.conv_layer1.pooled_out);
        self.rnn_layer2.calc(&self.rnn_layer1.bwd.output);
        let out = self.out_layer.calc(&self.rnn_layer2.bwd.output);


        out
    }
}


struct Conv33_64 {
}

impl ConvSizer for Conv33_64 {
  fn input_features() -> usize { 2 }
  fn output_features() -> usize { 64 }
  fn conv_filter_size() -> usize { 33 }
  fn sequence_size() -> usize { 3*STEP + 6*PAD }
  fn pool_kernel() -> usize { 3 }
}

struct GRU64 {
}

static mut JITTER64: *mut c_void = ptr::null_mut();
static mut SGEMM64: SgemmJitKernelT = None;

impl GRUSizer for GRU64 {
  fn sequence_size() -> usize { STEP + 2*PAD }
  fn output_features() -> usize { 64 }
  fn jitter() -> *mut c_void { unsafe { JITTER64 } }
  fn sgemm() -> SgemmJitKernelT { unsafe { SGEMM64 } }
}

struct Net64 {
    conv_layer1: ConvLayer<Conv33_64>,
    rnn_layer1: BiGRULayer<GRU64>,
    rnn_layer2: BiGRULayer<GRU64>,
    out_layer: OutLayer,
}

impl Net64 {
    fn new(filename: &str) -> Result<Net64, Box<dyn Error>> {
        let netfile = File::open(filename)?;
        let mut bufnetfile = BufReader::new(&netfile);
        let conv_layer1 = ConvLayer::new(&mut bufnetfile)?;
        let rnn_layer1 = BiGRULayer::new(&mut bufnetfile)?;
        let rnn_layer2 = BiGRULayer::new(&mut bufnetfile)?;
        let out_layer = OutLayer::new(&mut bufnetfile)?;
        Ok(Net64 {
            conv_layer1,
            rnn_layer1,
            rnn_layer2,
            out_layer,
        })
    }
}

impl Net for Net64 {
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

        self.conv_layer1.calc(&scaled_data);
        self.rnn_layer1.calc(&self.conv_layer1.pooled_out);
        self.rnn_layer2.calc(&self.rnn_layer1.bwd.output);
        let out = self.out_layer.calc(&self.rnn_layer2.bwd.output);

        out
    }
}

struct Conv33_80 {
}

impl ConvSizer for Conv33_80 {
  fn input_features() -> usize { 2 }
  fn output_features() -> usize { 80 }
  fn conv_filter_size() -> usize { 33 }
  fn sequence_size() -> usize { 3*STEP + 6*PAD }
  fn pool_kernel() -> usize { 3 }
}

struct GRU80 {
}

static mut JITTER80: *mut c_void = ptr::null_mut();
static mut SGEMM80: SgemmJitKernelT = None;

impl GRUSizer for GRU80 {
  fn sequence_size() -> usize { STEP + 2*PAD }
  fn output_features() -> usize { 80 }
  fn jitter() -> *mut c_void { unsafe { JITTER80 } }
  fn sgemm() -> SgemmJitKernelT { unsafe { SGEMM80 } }
}

struct Net80 {
    conv_layer1: ConvLayer<Conv33_80>,
    rnn_layer1: BiGRULayer<GRU80>,
    rnn_layer2: BiGRULayer<GRU80>,
    out_layer: OutLayer,
}

impl Net80 {
    fn new(filename: &str) -> Result<Net80, Box<dyn Error>> {
        let netfile = File::open(filename)?;
        let mut bufnetfile = BufReader::new(&netfile);
        let conv_layer1 = ConvLayer::new(&mut bufnetfile)?;
        let rnn_layer1 = BiGRULayer::new(&mut bufnetfile)?;
        let rnn_layer2 = BiGRULayer::new(&mut bufnetfile)?;
        let out_layer = OutLayer::new(&mut bufnetfile)?;
        Ok(Net80 {
            conv_layer1,
            rnn_layer1,
            rnn_layer2,
            out_layer,
        })
    }
}

impl Net for Net80 {
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

        self.conv_layer1.calc(&scaled_data);
        self.rnn_layer1.calc(&self.conv_layer1.pooled_out);
        self.rnn_layer2.calc(&self.rnn_layer1.bwd.output);
        let out = self.out_layer.calc(&self.rnn_layer2.bwd.output);

        out
    }
}

struct Conv33_96 {
}

impl ConvSizer for Conv33_96 {
  fn input_features() -> usize { 2 }
  fn output_features() -> usize { 96 }
  fn conv_filter_size() -> usize { 33 }
  fn sequence_size() -> usize { 3*STEP + 6*PAD }
  fn pool_kernel() -> usize { 3 }
}

struct GRU96 {
}

static mut JITTER96: *mut c_void = ptr::null_mut();
static mut SGEMM96: SgemmJitKernelT = None;

impl GRUSizer for GRU96 {
  fn sequence_size() -> usize { STEP + 2*PAD }
  fn output_features() -> usize { 96 }
  fn jitter() -> *mut c_void { unsafe { JITTER96 } }
  fn sgemm() -> SgemmJitKernelT { unsafe { SGEMM96 } }
}

struct Net96 {
    conv_layer1: ConvLayer<Conv33_96>,
    rnn_layer1: BiGRULayer<GRU96>,
    rnn_layer2: BiGRULayer<GRU96>,
    out_layer: OutLayer,
}

impl Net96 {
    fn new(filename: &str) -> Result<Net96, Box<dyn Error>> {
        let netfile = File::open(filename)?;
        let mut bufnetfile = BufReader::new(&netfile);
        let conv_layer1 = ConvLayer::new(&mut bufnetfile)?;
        let rnn_layer1 = BiGRULayer::new(&mut bufnetfile)?;
        let rnn_layer2 = BiGRULayer::new(&mut bufnetfile)?;
        let out_layer = OutLayer::new(&mut bufnetfile)?;
        Ok(Net96 {
            conv_layer1,
            rnn_layer1,
            rnn_layer2,
            out_layer,
        })
    }
}

impl Net for Net96 {
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

        self.conv_layer1.calc(&scaled_data);
        self.rnn_layer1.calc(&self.conv_layer1.pooled_out);
        self.rnn_layer2.calc(&self.rnn_layer1.bwd.output);
        let out = self.out_layer.calc(&self.rnn_layer2.bwd.output);

        out
    }
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
}

impl Net for NetBig {
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

        self.conv_layer1.calc(&scaled_data);
        self.rnn_layer1.calc(&self.conv_layer1.pooled_out);
        self.rnn_layer2.calc(&self.rnn_layer1.bwd.output);
        self.rnn_layer3.calc(&self.rnn_layer2.bwd.output);
        let out = self.out_layer.calc(&self.rnn_layer3.bwd.output);

        out
    }
}


#[pyclass]
struct Caller {
    net: Box<dyn Net>,
    beam_size: usize,
    beam_cut_threshold: f32
}

#[pymethods]
impl Caller {
    #[new]
    fn new(obj: &PyRawObject, net_type: &str, path: &str, beam_size: usize, beam_cut_threshold: f32) {
        if net_type == "256" {
            unsafe {
              if JITTER256 == ptr::null_mut() {
                initialize_jit256();
              }
            }
        } else if net_type == "56" {
            unsafe {
              if JITTER56 == ptr::null_mut() {
                initialize_jit56();
              }
            }
        } else if net_type == "64" {
            unsafe {
              if JITTER64 == ptr::null_mut() {
                initialize_jit64();
              }
            }
        } else if net_type == "80" {
            unsafe {
              if JITTER80 == ptr::null_mut() {
                initialize_jit80();
              }
            }
        } else if net_type == "96" {
            unsafe {
              if JITTER96 == ptr::null_mut() {
                initialize_jit96();
              }
            }
        } else {
            unsafe {
              if JITTER48 == ptr::null_mut() {
                initialize_jit48();
              }
            }
        }
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
        obj.init({
            Caller {
                net, beam_size, beam_cut_threshold
            }
        })
    }
    
    fn call_raw_signal(&mut self, raw_data: Vec<f32>) -> String {
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

        let mut result = stack(
            Axis(0),
            &(to_stack.iter().map(|x| x.view()).collect::<Vec<_>>()),
        ).unwrap();

        if self.beam_size > 1 {
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
        } else {
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
}

#[pyfunction]
fn beam_search_py(result: &PyArray2<f32>, beam_size: usize, beam_cut_threshold: f32) -> String {
    beam_search(&result.as_array(), beam_size, beam_cut_threshold)
}

fn beam_search<D: Data<Elem=f32>>(result: &ArrayBase<D, Ix2>, beam_size: usize, beam_cut_threshold: f32) -> String {
    let alphabet: Vec<char> = "NACGT".chars().collect();
    // (base, what)
    let mut beam_prevs = vec![(0, 0)];
    let mut beam_forward: Vec<[i32; 4]> = vec![[-1, -1, -1, -1]];
    let mut cur_probs = vec![(0i32, 0.0, 1.0)];
    let mut new_probs = Vec::new();
    
    for pr in result.slice(s![..;-1, ..]).outer_iter() {
        new_probs.clear();

        for &(beam, base_prob, n_prob) in &cur_probs {
            // add N to beam
            if pr[0] > beam_cut_threshold {
                new_probs.push((beam, 0.0, (n_prob + base_prob) * pr[0]));
            }

            for b in 1..5 {
                if pr[b] < beam_cut_threshold {
                    continue
                }
                if b == beam_prevs[beam as usize].0 {
                    new_probs.push((beam, base_prob * pr[b], 0.0));
                    let mut new_beam = beam_forward[beam as usize][b-1];
                    if new_beam == -1 {
                        new_beam = beam_prevs.len() as i32;
                        beam_prevs.push((b, beam));
                        beam_forward[beam as usize][b-1] = new_beam;
                        beam_forward.push([-1, -1, -1, -1]);
                    }

                    new_probs.push((new_beam, n_prob * pr[b], 0.0));

                } else {
                    let mut new_beam = beam_forward[beam as usize][b-1];
                    if new_beam == -1 {
                        new_beam = beam_prevs.len() as i32;
                        beam_prevs.push((b, beam));
                        beam_forward[beam as usize][b-1] = new_beam;
                        beam_forward.push([-1, -1, -1, -1]);
                    }

                    new_probs.push((new_beam, (base_prob + n_prob) * pr[b], 0.0));
                }
            }
        }
        std::mem::swap(&mut cur_probs, &mut new_probs);

        cur_probs.sort_by_key(|x| x.0);
        let mut last_key: i32 = -1;
        let mut last_key_pos = 0;
        for i in 0..cur_probs.len() {
            if cur_probs[i].0 == last_key {
                cur_probs[last_key_pos].1 = cur_probs[last_key_pos].1 + cur_probs[i].1;
                cur_probs[last_key_pos].2 = cur_probs[last_key_pos].2 +cur_probs[i].2;
                cur_probs[i].0 = -1;
            } else {
                last_key_pos = i;
                last_key = cur_probs[i].0;
            }
        }

        cur_probs.retain(|x| x.0 != -1);
        cur_probs.sort_by(|a, b| (b.1 + b.2).partial_cmp(&(a.1 + a.2)).unwrap());
        cur_probs.truncate(beam_size);
        let top = cur_probs[0].1 + cur_probs[0].2;
        for mut x in &mut cur_probs {
            x.1 /= top;
            x.2 /= top;
        }
    }

    let mut out = String::new();
    let mut beam = cur_probs[0].0;
    while beam != 0 {
        out.push(alphabet[beam_prevs[beam as usize].0]);
        beam = beam_prevs[beam as usize].1;
    }
    out
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

fn initialize_jit56() {
    unsafe {
        JITTER56 = ptr::null_mut();
        // TODO: check
        let status = mkl_cblas_jit_create_sgemm(
            &mut JITTER56,
            101,
            111,
            111,
            1,
            56 * 3,
            56,
            1.0,
            56,
            56 * 3,
            0.0,
            56 * 3,
        );

        SGEMM56 = mkl_jit_get_sgemm_ptr(JITTER56);
    }
}

fn initialize_jit64() {
    unsafe {
        JITTER64 = ptr::null_mut();
        // TODO: check
        let status = mkl_cblas_jit_create_sgemm(
            &mut JITTER64,
            101,
            111,
            111,
            1,
            64 * 3,
            64,
            1.0,
            64,
            64 * 3,
            0.0,
            64 * 3,
        );

        SGEMM64 = mkl_jit_get_sgemm_ptr(JITTER64);
    }
}

fn initialize_jit80() {
    unsafe {
        JITTER80 = ptr::null_mut();
        // TODO: check
        let status = mkl_cblas_jit_create_sgemm(
            &mut JITTER80,
            101,
            111,
            111,
            1,
            80 * 3,
            80,
            1.0,
            80,
            80 * 3,
            0.0,
            80 * 3,
        );

        SGEMM80 = mkl_jit_get_sgemm_ptr(JITTER80);
    }
}


fn initialize_jit96() {
    unsafe {
        JITTER96 = ptr::null_mut();
        // TODO: check
        let status = mkl_cblas_jit_create_sgemm(
            &mut JITTER96,
            101,
            111,
            111,
            1,
            96 * 3,
            96,
            1.0,
            96,
            96 * 3,
            0.0,
            96 * 3,
        );

        SGEMM96 = mkl_jit_get_sgemm_ptr(JITTER96);
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
    m.add_wrapped(wrap_pyfunction!(beam_search_py))?;

    Ok(())
}
