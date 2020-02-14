/// Raw transmutation from `u32`.
///
/// Converts the given `u32` containing the float's raw memory representation into the `f32` type.
/// Similar to `f32::from_bits` but even more raw.
#[inline]
pub fn from_bits(x: u32) -> f32 {
    unsafe { ::std::mem::transmute::<u32, f32>(x) }
}

/// Raises 2 to a floating point power.
//#[inline]
//pub fn pow2(p: f32) -> f32 {
//    let v = (8388608.0_f32 * p + 1064872507.1541044_f32) as u32;
//    from_bits(v)
//}

/// Exponential function.
#[inline]
pub fn exp(p: f32) -> f32 {
    //pow2(1.442695040_f32 * p)
    let v = (12102203.15410432_f32 * p + 1064872507.1541044_f32) as u32;
    from_bits(v)

}

#[inline]
pub fn exp_m2(p: f32) -> f32 {
    //pow2(-2.0 * 1.442695040_f32 * p)
    let v = (-24204406.30820864_f32 * p + 1064872507.1541044_f32) as u32;
    from_bits(v)

}

/// Sigmoid function.
#[inline]
pub fn approx_nsigmoid(x: f32) -> f32 {
    1.0_f32 / (1.0_f32 + exp(x))
}

/// Hyperbolic tangent function.
#[inline]
pub fn approx_tanh(p: f32) -> f32 {
    -1.0_f32 + 2.0_f32 / (1.0_f32 + exp_m2(p))
}
