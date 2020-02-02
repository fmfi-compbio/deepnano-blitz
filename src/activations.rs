use std::mem;

#[inline]
pub fn inv_sqrtf32(x: f32) -> f32 {
     // Magic number based on Chris Lomont work:
    // const MAGIC_U32: u32 = 0x5f375a86;
    // The Original Magic Number:
    // const MAGIC_32: u32 = 0x5f3759df;
    const threehalfs: f32 = 1.5f32;
    let x2: f32 = x * 0.5f32;
    let mut i: u32 = unsafe { mem::transmute(x) }; // evil floating point bit level hacking
    i = 0x5f375a86 - (i >> 1);                        // what the fuck?
    let y: f32 = unsafe { mem::transmute(i) };
    let y  = y * ( threehalfs - ( x2 * y * y ) );     // 1st iteration
//		y  = y * ( threehalfs - ( x2 * y * y ) );       // 2nd iteration, this can be removed
    return y;
}

#[inline]
pub fn tanh(x: f32) -> f32 {
    return x * inv_sqrtf32(1.0f32+x*x)
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    0.5f32 + 0.5f32 * tanh(x)
}
