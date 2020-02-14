use ndarray::linalg::general_mat_mul;
use ndarray::{Array, Ix1, Ix2};
use crate::matrix_load::*;
use crate::approx::*;
use std::io::BufRead;
use std::error::Error;
use std::marker::PhantomData;

pub trait ConvSizer {
  fn output_features() -> usize;
  fn conv_filter_size() -> usize;
  fn sequence_size() -> usize;
  fn input_features() -> usize;
  fn pool_kernel() -> usize;
}

pub struct ConvLayer<CS: ConvSizer> {
    w: Array<f32, Ix2>,
    b: Array<f32, Ix1>,
    im2col: Array<f32, Ix2>,
    convout: Array<f32, Ix2>,
    pub pooled_out: Array<f32, Ix2>,
    phantom: PhantomData<CS>
}

impl<CS: ConvSizer> ConvLayer<CS> {
    pub fn new<R: BufRead>(f: &mut R) -> Result<ConvLayer<CS>, Box<dyn Error>> {
        Ok(ConvLayer {
            w: load2dmatrix(f)?,
            b: load1dmatrix(f)?,
            im2col: align2d(Array::from_elem((CS::sequence_size(), CS::conv_filter_size() * CS::input_features()), 0.0)),
            convout: align2d(Array::from_elem((CS::sequence_size(), CS::output_features()), 0.0)),
            pooled_out: align2d(Array::from_elem((CS::sequence_size() / CS::pool_kernel(), CS::output_features()), 0.0)),
            phantom: PhantomData
        })
    }
    
    fn max_pool(input: &Array<f32, Ix2>, out: &mut Array<f32, Ix2>) {

        unsafe {
            let out_ptr = out.as_mut_ptr();
            let mut out_offset = 0;
            let in_ptr = input.as_ptr();
            let mut in_offset = 0;
            for _i in 0..out.shape()[0] {
                for _j in 0..out.shape()[1] {
                    let mut m = *in_ptr.offset(in_offset);
                    let mut in_offset2 = in_offset;
                    for _k in 1..CS::pool_kernel() {
                        in_offset2 += out.shape()[1] as isize;
                        m = m.max(*in_ptr.offset(in_offset2));
                    }

                    *out_ptr.offset(out_offset) = m;
                    out_offset += 1;
                    in_offset += 1;
                }
                in_offset += 2 * out.shape()[1] as isize;
            }
        }
    }


    pub fn calc(&mut self, input: &Array<f32, Ix2>) {
        unsafe {
            let input_ptr = input.as_ptr();
            let im2col_ptr = self.im2col.as_mut_ptr();
            for i in 0..input.shape()[0] {
                let mut imoffset = (i as isize + (CS::conv_filter_size() / 2) as isize) * (CS::conv_filter_size() * CS::input_features()) as isize;
                for _j in 0..CS::conv_filter_size() as isize {
                    if imoffset >= 0 && imoffset < self.im2col.len() as isize {
                        for f in 0..CS::input_features() as isize {
                          *im2col_ptr.offset(imoffset + f) = *input_ptr.offset(i as isize * 2 + f);
                        }
                    }
                    imoffset -= ((CS::conv_filter_size() * CS::input_features()) - CS::input_features()) as isize;
                }
            }
        }
        general_mat_mul(1.0, &self.im2col, &self.w, 0.0, &mut self.convout);
        let mut pooled = Self::max_pool(&self.convout, &mut self.pooled_out);
        self.pooled_out += &self.b;
        unsafe {
            let ptr = self.pooled_out.as_mut_ptr();
            for i in 0..self.pooled_out.len() as isize {
                *ptr.offset(i) = approx_tanh(*ptr.offset(i))
            }
        }
    }
}

