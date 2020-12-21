use ndarray::{Ix2, ArrayBase, Data};

pub fn beam_search<D: Data<Elem=f32>>(result: &ArrayBase<D, Ix2>, beam_size: usize, beam_cut_threshold: f32) -> (String,String) {
    let alphabet: Vec<char> = "NACGT".chars().collect();
    // (base, what)
    let mut beam_prevs = vec![(0, 0)];
    let mut beam_max_p = vec![(0.0f32)];
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
                    beam_max_p[beam as usize] = beam_max_p[beam as usize].max(pr[b]);
                    let mut new_beam = beam_forward[beam as usize][b-1];
                    if new_beam == -1 {
                        new_beam = beam_prevs.len() as i32;
                        beam_prevs.push((b, beam));
                        beam_max_p.push(pr[b]);
                        beam_forward[beam as usize][b-1] = new_beam;
                        beam_forward.push([-1, -1, -1, -1]);
                    }

                    new_probs.push((new_beam, n_prob * pr[b], 0.0));
                    beam_max_p[new_beam as usize] = beam_max_p[new_beam as usize].max(pr[b]);

                } else {
                    let mut new_beam = beam_forward[beam as usize][b-1];
                    if new_beam == -1 {
                        new_beam = beam_prevs.len() as i32;
                        beam_prevs.push((b, beam));
                        beam_max_p.push(pr[b]);
                        beam_forward[beam as usize][b-1] = new_beam;
                        beam_forward.push([-1, -1, -1, -1]);
                    }

                    new_probs.push((new_beam, (base_prob + n_prob) * pr[b], 0.0));
                    beam_max_p[new_beam as usize] = beam_max_p[new_beam as usize].max(pr[b]);
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
    let mut out_p = String::new();
    let mut beam = cur_probs[0].0;
    while beam != 0 {
        out.push(alphabet[beam_prevs[beam as usize].0]);
        out_p.push(prob_to_str(beam_max_p[beam as usize]));
        beam = beam_prevs[beam as usize].1;
    }
    (out, out_p)
}

fn prob_to_str(x: f32) -> char {
    let q = (-(1.0-x).log10()*10.0) as u32 + 33;
    std::char::from_u32(q).unwrap()
}