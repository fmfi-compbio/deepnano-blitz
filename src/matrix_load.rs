use std::io::BufRead;
use ndarray::{Array, Ix1, Ix2};
use std::error::Error;


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
    Ok(Array::from_shape_vec((shape[0], shape[1]), data)?)
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
    Ok(Array::from_shape_vec(shape[0], data)?)
}
