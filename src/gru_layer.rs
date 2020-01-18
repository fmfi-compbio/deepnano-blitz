use ndarray::{stack, Array, Axis, Ix1, Ix2};
use std::error::Error;
use std::io::BufRead;
use ndarray::linalg::general_mat_mul;
use crate::matrix_load::*;
use libc::c_void;
use std::marker::PhantomData;

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
        let wourn = stack!(Axis(1), -wou, -wor, woo); //.t().to_owned();
        let wiurn = stack!(Axis(1), -wiu, -wir, wio);
        let biur = stack!(Axis(0), -biu - bou, -bir - bor);
        Ok(GRULayer {
            wourn: wourn,
            wiurn: wiurn,
            biur: biur,
            bio: bio,
            boo: boo,
            input_proc: Array::from_elem((GS::sequence_size(), GS::output_features() * 3), 0.0),
            phantom: PhantomData
        })

        // TODO: assert sizes
    }

    pub fn calc(&mut self, input: &Array<f32, Ix2>) -> Array<f32, Ix2> {
        let mut state = Array::from_elem(GS::output_features(), 0.0f32);
        let mut output = Array::from_elem((GS::sequence_size(), GS::output_features()), 0.0f32);
        general_mat_mul(1.0, &input, &self.wiurn, 0.0, &mut self.input_proc);

        let mut state_proc = Array::from_elem(GS::output_features() * 3, 0.0f32);
        let mut new_val = Array::from_elem(GS::output_features(), 0.0f32);

        let n_steps = self.input_proc.shape()[0];

        for (num, sample) in self.input_proc.outer_iter().enumerate() {
            unsafe {
                let old_st_ptr = if num == 0 {
                    state.as_mut_ptr()
                } else {
                    output.as_mut_ptr().offset(((n_steps - num) as isize) * GS::output_features() as isize)
                };
                GS::sgemm().unwrap()(
                    GS::jitter(),
                    old_st_ptr,
                    self.wourn.as_mut_ptr(),
                    state_proc.as_mut_ptr(),
                );
            }
            {
                unsafe {
                    let ptr = state_proc.as_mut_ptr();
                    let sptr = sample.as_ptr();
                    let bptr = self.biur.as_ptr();
		    for i in 0..2*GS::output_features() as isize {
                        *ptr.offset(i) = 1.0 / (1.0 + fastapprox::faster::exp(*ptr.offset(i) + *sptr.offset(i) + *bptr.offset(i)));
                    }
                }
            }

            unsafe {
                let nvptr = new_val.as_mut_ptr();
                let ptr = state_proc.as_ptr();
                let sptr = sample.as_ptr();
                let stptr = output.as_mut_ptr().offset((n_steps - num - 1) as isize * GS::output_features() as isize);
                let biptr = self.bio.as_ptr();
                let boptr = self.boo.as_ptr();
                let old_st_ptr = if num == 0 {
                    state.as_mut_ptr()
                } else {
                    output.as_mut_ptr().offset(((n_steps - num) as isize) * GS::output_features() as isize)
                };

                for i in 0..GS::output_features() as isize {
                    *nvptr.offset(i) = (*ptr.offset(2 * GS::output_features() as isize + i) + *boptr.offset(i))
                        * *ptr.offset(GS::output_features() as isize + i)
                        + *sptr.offset(2*GS::output_features() as isize + i)
                        + *biptr.offset(i);
                }

		for i in 0..GS::output_features() as isize {
                    *stptr.offset(i) = *old_st_ptr.offset(i) * *ptr.offset(i)
                            + (1.0 - *ptr.offset(i)) * fastapprox::faster::tanh(*nvptr.offset(i));
                }
            }
        }
        output
    }
}

pub struct BiGRULayer<GS: GRUSizer> {
    fwd: GRULayer<GS>,
    bwd: GRULayer<GS>,
}

impl<GS: GRUSizer> BiGRULayer<GS> {
    pub fn new<R: BufRead>(f: &mut R) -> Result<BiGRULayer<GS>, Box<dyn Error>> {
        Ok(BiGRULayer {
            fwd: GRULayer::new(f)?,
            bwd: GRULayer::new(f)?,
        })
    }

    pub fn calc(&mut self, input: &Array<f32, Ix2>) -> Array<f32, Ix2> {
        let fwd_res = self.fwd.calc(input);
        let bwd_res = self.bwd.calc(&fwd_res);

        return bwd_res;
    }
}
