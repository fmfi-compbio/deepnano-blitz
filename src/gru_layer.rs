use ndarray::{stack, Array, Axis, Ix1, Ix2};
use std::error::Error;
use std::io::BufRead;
use ndarray::linalg::general_mat_mul;
use crate::matrix_load::*;
use libc::c_void;
use std::marker::PhantomData;
use crate::activations;

pub type SgemmJitKernelT =
    Option<unsafe extern "C" fn(arg1: *mut c_void, arg2: *mut f32, arg3: *mut f32, arg4: *mut f32)>;

pub trait GRUSizer {
  fn sequence_size() -> usize;
  fn output_features() -> usize;
  fn jitter() -> *mut c_void;
  fn sgemm() -> SgemmJitKernelT;
}

pub struct GRULayer<GS: GRUSizer> {
    wourn: Array<f32, Ix2>,
    wiurn: Array<f32, Ix2>,
    biur: Array<f32, Ix1>,
    bio: Array<f32, Ix1>,
    boo: Array<f32, Ix1>,
    input_proc: Array<f32, Ix2>, //    h0: Array<f32, Ix1>,
    state: Array<f32, Ix1>,
    pub output: Array<f32, Ix2>,
    state_proc: Array<f32, Ix1>,
    phantom: PhantomData<GS>
}

impl<GS: GRUSizer> GRULayer<GS> {
    pub fn new<R: BufRead>(f: &mut R) -> Result<GRULayer<GS>, Box<dyn Error>> {
        let wio = load2dmatrix(f)?;
        let woo = load2dmatrix(f)?;
        let bio = load1dmatrix(f)?;
        let boo = load1dmatrix(f)?;
        let wir = load2dmatrix(f)?;
        let wiu = load2dmatrix(f)?;
        let wor = load2dmatrix(f)?;
        let wou = load2dmatrix(f)?;
        let bir = load1dmatrix(f)?;
        let bor = load1dmatrix(f)?;
        let biu = load1dmatrix(f)?;
        let bou = load1dmatrix(f)?;
        let wourn = align2d(stack!(Axis(1), -wou, -wor, woo)); //.t().to_owned();
        let wiurn = align2d(stack!(Axis(1), -wiu, -wir, wio));
        let biur = align1d(stack!(Axis(0), -biu - bou, -bir - bor));
        let input_proc = align2d(Array::from_elem((GS::sequence_size(), GS::output_features() * 3), 0.0));
        let state = align1d(Array::from_elem(GS::output_features(), 0.0));
        let state_proc = align1d(Array::from_elem(GS::output_features() * 4, 0.0));
        let output = align2d(Array::from_elem((GS::sequence_size(), GS::output_features()), 0.0));

        Ok(GRULayer {
            wourn: wourn,
            wiurn: wiurn,
            biur: biur,
            bio: bio,
            boo: boo,
            input_proc,
            state,
            output,
            state_proc,
            phantom: PhantomData
        })

        // TODO: assert sizes
    }

    pub fn calc(&mut self, input: &Array<f32, Ix2>) {
//        let mut state = Array::from_elem(GS::output_features(), 0.0f32);
//        let mut output = Array::from_elem((GS::sequence_size(), GS::output_features()), 0.0f32);
        general_mat_mul(1.0, &input, &self.wiurn, 0.0, &mut self.input_proc);

//        let mut state_proc = Array::from_elem(GS::output_features() * 3, 0.0f32);
//        let mut new_val = Array::from_elem(GS::output_features(), 0.0f32);

        let n_steps = self.input_proc.shape()[0];

        for (num, sample) in self.input_proc.outer_iter().enumerate() {
            unsafe {
                let old_st_ptr = if num == 0 {
                    self.state.as_mut_ptr()
                } else {
                    self.output.as_mut_ptr().offset(((n_steps - num) as isize) * GS::output_features() as isize)
                };
                GS::sgemm().unwrap()(
                    GS::jitter(),
                    old_st_ptr,
                    self.wourn.as_mut_ptr(),
                    self.state_proc.as_mut_ptr(),
                );
            }
            {
                unsafe {
                    let ptr = self.state_proc.as_mut_ptr();
                    let sptr = sample.as_ptr();
                    let bptr = self.biur.as_ptr();
		    for i in 0..2*GS::output_features() as isize {
                        *ptr.offset(i) = crate::activations::sigmoid(-(*ptr.offset(i) + *sptr.offset(i) + *bptr.offset(i)));
                    }
                }
            }

            unsafe {
                let ptr = self.state_proc.as_mut_ptr();
                let nvptr = ptr.offset(3*GS::output_features() as isize);
                let sptr = sample.as_ptr();
                let stptr = self.output.as_mut_ptr().offset((n_steps - num - 1) as isize * GS::output_features() as isize);
                let biptr = self.bio.as_ptr();
                let boptr = self.boo.as_ptr();
                let old_st_ptr = if num == 0 {
                    self.state.as_mut_ptr()
                } else {
                    self.output.as_mut_ptr().offset(((n_steps - num) as isize) * GS::output_features() as isize)
                };

                for i in 0..GS::output_features() as isize {
                    *nvptr.offset(i) = (*ptr.offset(2 * GS::output_features() as isize + i) + *boptr.offset(i))
                        * *ptr.offset(GS::output_features() as isize + i)
                        + *sptr.offset(2*GS::output_features() as isize + i)
                        + *biptr.offset(i);
                }

		for i in 0..GS::output_features() as isize {
                    *stptr.offset(i) = *old_st_ptr.offset(i) * *ptr.offset(i)
                            + (1.0 - *ptr.offset(i)) * crate::activations::tanh(*nvptr.offset(i));
                }
            }
        }
    }
}

pub struct BiGRULayer<GS: GRUSizer> {
    fwd: GRULayer<GS>,
    pub bwd: GRULayer<GS>,
}

impl<GS: GRUSizer> BiGRULayer<GS> {
    pub fn new<R: BufRead>(f: &mut R) -> Result<BiGRULayer<GS>, Box<dyn Error>> {
        Ok(BiGRULayer {
            fwd: GRULayer::new(f)?,
            bwd: GRULayer::new(f)?,
        })
    }

    pub fn calc(&mut self, input: &Array<f32, Ix2>) {
        self.fwd.calc(input);
        self.bwd.calc(&self.fwd.output);
    }
}
