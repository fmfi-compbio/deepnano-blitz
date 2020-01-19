#!/usr/bin/env python

from ont_fast5_api.fast5_interface import get_fast5_file
import argparse
import os
import numpy as np
import datetime
import deepnano2
from multiprocessing import Pool
import torch
from deepnano2.gpu_model import Net

step = 550
pad = 25
batch_size = 1024
reads_in_group = 100

def med_mad(x, factor=1.4826):
    """
    Calculate signal median and median absolute deviation
    """
    med = np.median(x)
    mad = np.median(np.absolute(x - med)) * factor
    return med, mad

def rescale_signal(signal):
    signal = signal.astype(np.float32)
    med, mad = med_mad(signal)
    signal -= med
    signal /= mad       
    return np.clip(signal, -2.5, 2.5)

def call_group(group):
    chunks = []
    read_lens = []
    for read_id, signal in group:
        rl = 0
        for i in range(0, len(signal), 3*step):
            if i + 3*step + 6*pad > len(signal):
                break
            part = np.array(signal[i:i+3*step+6*pad])    
            chunks.append(np.vstack([part, part * part]).T)
            rl += 1
        read_lens.append((read_id, rl))

    chunks = np.stack(chunks)
    outputs = []
    for i in range(0, len(chunks), batch_size):
        print("bs", len(chunks[i:i+batch_size]), datetime.datetime.now())
        net_result = model(torch.Tensor(chunks[i:i+batch_size]).cuda()).detach().cpu().numpy()

        for row in net_result:
            outputs.append(row[pad:-pad])

    last_f = 0
    alph = np.array(["N", "A", "C", "G", "T"])
    out = []
    for read_id, rl in read_lens:
        seq = []
        last = 47
        stacked = np.vstack(outputs[last_f:last_f+rl])
        am = np.argmax(stacked, axis=1)
        selection = np.ones(len(am), dtype=bool)
        selection[1:] = am[1:] != am[:-1]
        selection &= am != 0
        seq = "".join(alph[am[selection]])

        out.append((read_id, seq))
        last_f += rl
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fast caller for ONT reads')

    parser.add_argument('--directory', type=str, nargs='*', help='One or more directories with reads')
    parser.add_argument('--reads', type=str, nargs='*', help='One or more read files')
    parser.add_argument("--output", type=str, required=True, help="Output FASTA file name")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads for basecalling")
    parser.add_argument("--weights", type=str, default=None, help="Path to network weights")
    parser.add_argument("--network-type", choices=["fast", "accurate"], default="fast")
    parser.add_argument("--beam-size", type=int, default=None, help="Beam size (defaults 5 for fast and 20 for accurate. Use 1 to disable.")
    parser.add_argument("--beam-cut-threshold", type=float, default=None, help="Threshold for creating beams (higher means faster beam search, but smaller accuracy)")

    args = parser.parse_args()

    assert args.threads >= 1

    files = args.reads if args.reads else []
    if args.directory:
        for directory_name in args.directory:
            files += [os.path.join(directory_name, fn) for fn in os.listdir(directory_name)]

    if len(files) == 0:
        print("Zero input reads, nothing to do.")
        sys.exit()

    weights = os.path.join(deepnano2.__path__[0], "weights", "weightsbig520.pt")
    
    torch.set_grad_enabled(False)
    model = Net()
    model.load_state_dict(torch.load(weights))
    model.eval()
    model.cuda()

    fout = open(args.output, "w")

    done = 0

    group = []
    print("start", datetime.datetime.now())
    for fn in files:
        try:
            with get_fast5_file(fn, mode="r") as f5:
                for read in f5.get_reads():
                    group.append((read.get_read_id(), rescale_signal(read.get_raw_data())))
                    if len(group) >= reads_in_group:
                        for read_id, basecall in call_group(group):
                            print(">%s" % read_id, file=fout)
                            print(basecall, file=fout)
                        group = []
        except OSError:
            # TODO show something here
            pass
    if len(group) > 0:
        for read_id, basecall in call_group(group):
            print(">%s" % read_id, file=fout)
            print(basecall, file=fout)

    print("fin", datetime.datetime.now())
