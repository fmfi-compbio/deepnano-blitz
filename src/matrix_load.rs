use std::io::BufRead;
use ndarray::{Array, Ix1, Ix2};
use std::error::Error;

const ALIGN: usize = 64;

pub fn align2d(input: Array<f32, Ix2>) -> Array<f32, Ix2> {
    unsafe {
        let optr = std::alloc::alloc(std::alloc::Layout::from_size_align(input.len() * 4, ALIGN).unwrap()) as *mut f32;
        let ov = Vec::from_raw_parts(optr, input.len(), input.len());
        let mut res = Array::from_shape_vec_unchecked(input.shape(), ov);
        res.assign(&input);
        res.into_dimensionality::<Ix2>().unwrap()
    } 
}

pub fn align1d(input: Array<f32, Ix1>) -> Array<f32, Ix1> {
    unsafe {
        let optr = std::alloc::alloc(std::alloc::Layout::from_size_align(input.len() * 4, ALIGN).unwrap()) as *mut f32;
        let ov = Vec::from_raw_parts(optr, input.len(), input.len());
        let mut res = Array::from_shape_vec_unchecked(input.shape(), ov);
        res.assign(&input);
        res.into_dimensionality::<Ix1>().unwrap()
    } 
}

pub fn load2dmatrix<R: BufRead>(f: &mut R) -> Result<Array<f32, Ix2>, Box<dyn Error>> {
    let mut line = String::new();
    f.read_line(&mut line)?;
    line.pop();
    let shape = line
        .split(" ")
        .map(|x| x.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    line.clear();
    f.read_line(&mut line)?;
    line.pop();

    let data = line
        .trim()
        .split(" ")
        .map(|x| x.parse::<f32>())
        .collect::<Result<Vec<_>, _>>()?;
    Ok(align2d(Array::from_shape_vec((shape[0], shape[1]), data)?))
}

pub fn load1dmatrix<R: BufRead>(f: &mut R) -> Result<Array<f32, Ix1>, Box<dyn Error>> {
    let mut line = String::new();
    f.read_line(&mut line)?;
    line.pop();
    let shape = line
        .split(" ")
        .map(|x| x.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    line.clear();
    f.read_line(&mut line)?;
    line.pop();

    let data = line
        .trim()
        .split(" ")
        .map(|x| x.parse::<f32>())
        .collect::<Result<Vec<_>, _>>()?;
    Ok(align1d(Array::from_shape_vec(shape[0], data)?))
}
