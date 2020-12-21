use crate::matrix_load::*;
use crate::conv_layer::*;
use crate::gru_layer::*;

use libc::{c_void};
use ndarray::{Array, Ix1, Ix2};
use std::error::Error;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::ptr;

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

pub trait Net {
    fn predict(&mut self, chunk: &[f32]) -> Array<f32, Ix2>;
}

pub struct NetSmall {
    conv_layer1: ConvLayer<Conv33_48>,
    rnn_layer1: BiGRULayer<GRU48>,
    rnn_layer2: BiGRULayer<GRU48>,
    out_layer: OutLayer,
}

impl NetSmall {
    pub fn new(filename: &str) -> Result<NetSmall, Box<dyn Error>> {
        let netfile = File::open(filename)?;
        let mut bufnetfile = BufReader::new(&netfile);
        let conv_layer1 = ConvLayer::new(&mut bufnetfile)?;
        let rnn_layer1 = BiGRULayer::new(&mut bufnetfile)?;
        let rnn_layer2 = BiGRULayer::new(&mut bufnetfile)?;
        let out_layer = OutLayer::new(&mut bufnetfile)?;
        
        unsafe {
            if JITTER48 == ptr::null_mut() {
                initialize_jit48();
            }
        }
        
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

pub struct Net56 {
    conv_layer1: ConvLayer<Conv33_56>,
    rnn_layer1: BiGRULayer<GRU56>,
    rnn_layer2: BiGRULayer<GRU56>,
    out_layer: OutLayer,
}

impl Net56 {
    pub fn new(filename: &str) -> Result<Net56, Box<dyn Error>> {
        let netfile = File::open(filename)?;
        let mut bufnetfile = BufReader::new(&netfile);
        let conv_layer1 = ConvLayer::new(&mut bufnetfile)?;
        let rnn_layer1 = BiGRULayer::new(&mut bufnetfile)?;
        let rnn_layer2 = BiGRULayer::new(&mut bufnetfile)?;
        let out_layer = OutLayer::new(&mut bufnetfile)?;

        unsafe {
            if JITTER56 == ptr::null_mut() {
            initialize_jit56();
            }
        }
            
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

pub struct Net64 {
    conv_layer1: ConvLayer<Conv33_64>,
    rnn_layer1: BiGRULayer<GRU64>,
    rnn_layer2: BiGRULayer<GRU64>,
    out_layer: OutLayer,
}

impl Net64 {
    pub fn new(filename: &str) -> Result<Net64, Box<dyn Error>> {
        let netfile = File::open(filename)?;
        let mut bufnetfile = BufReader::new(&netfile);
        let conv_layer1 = ConvLayer::new(&mut bufnetfile)?;
        let rnn_layer1 = BiGRULayer::new(&mut bufnetfile)?;
        let rnn_layer2 = BiGRULayer::new(&mut bufnetfile)?;
        let out_layer = OutLayer::new(&mut bufnetfile)?;

        unsafe {
            if JITTER64 == ptr::null_mut() {
            initialize_jit64();
            }
        }

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

pub struct Net80 {
    conv_layer1: ConvLayer<Conv33_80>,
    rnn_layer1: BiGRULayer<GRU80>,
    rnn_layer2: BiGRULayer<GRU80>,
    out_layer: OutLayer,
}

impl Net80 {
    pub fn new(filename: &str) -> Result<Net80, Box<dyn Error>> {
        let netfile = File::open(filename)?;
        let mut bufnetfile = BufReader::new(&netfile);
        let conv_layer1 = ConvLayer::new(&mut bufnetfile)?;
        let rnn_layer1 = BiGRULayer::new(&mut bufnetfile)?;
        let rnn_layer2 = BiGRULayer::new(&mut bufnetfile)?;
        let out_layer = OutLayer::new(&mut bufnetfile)?;

        unsafe {
            if JITTER80 == ptr::null_mut() {
              initialize_jit80();
            }
          }

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

pub struct Net96 {
    conv_layer1: ConvLayer<Conv33_96>,
    rnn_layer1: BiGRULayer<GRU96>,
    rnn_layer2: BiGRULayer<GRU96>,
    out_layer: OutLayer,
}

impl Net96 {
    pub fn new(filename: &str) -> Result<Net96, Box<dyn Error>> {
        let netfile = File::open(filename)?;
        let mut bufnetfile = BufReader::new(&netfile);
        let conv_layer1 = ConvLayer::new(&mut bufnetfile)?;
        let rnn_layer1 = BiGRULayer::new(&mut bufnetfile)?;
        let rnn_layer2 = BiGRULayer::new(&mut bufnetfile)?;
        let out_layer = OutLayer::new(&mut bufnetfile)?;

        unsafe {
            if JITTER96 == ptr::null_mut() {
                initialize_jit96();
            }
        }          

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

pub struct NetBig {
    conv_layer1: ConvLayer<Conv33_256>,
    rnn_layer1: BiGRULayer<GRU256>,
    rnn_layer2: BiGRULayer<GRU256>,
    rnn_layer3: BiGRULayer<GRU256>,
    out_layer: OutLayer,
}

impl NetBig {
    pub fn new(filename: &str) -> Result<NetBig, Box<dyn Error>> {
        let netfile = File::open(filename)?;
        let mut bufnetfile = BufReader::new(&netfile);
        let conv_layer1 = ConvLayer::new(&mut bufnetfile)?;
        let rnn_layer1 = BiGRULayer::new(&mut bufnetfile)?;
        let rnn_layer2 = BiGRULayer::new(&mut bufnetfile)?;
        let rnn_layer3 = BiGRULayer::new(&mut bufnetfile)?;
        let out_layer = OutLayer::new(&mut bufnetfile)?;
                      
        unsafe {
            if JITTER256 == ptr::null_mut() {
                initialize_jit256();
            }
        }

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


fn initialize_jit48() {
    unsafe {
        JITTER48 = ptr::null_mut();
        // TODO: check
        let _status = mkl_cblas_jit_create_sgemm(
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
        let _status = mkl_cblas_jit_create_sgemm(
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
        let _status = mkl_cblas_jit_create_sgemm(
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
        let _status = mkl_cblas_jit_create_sgemm(
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
        let _status = mkl_cblas_jit_create_sgemm(
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
        let _status = mkl_cblas_jit_create_sgemm(
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
