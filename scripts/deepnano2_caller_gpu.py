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
batch_size = 512

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

def call_signal(signal):
    chunks = []
    for i in range(0, len(signal), 3*step):
        if i + 3*step + 6*pad > len(signal):
            break
        part = np.array(signal[i:i+3*step+6*pad])    
        chunks.append(np.vstack([part, part * part]).T)
    chunks = np.stack(chunks)

    outputs = []
    for i in range(0, len(chunks), batch_size):
        net_result = model(torch.Tensor(chunks[i:i+batch_size]).cuda()).detach().cpu().numpy()

        for row in net_result:
            outputs.append(row[pad:-pad])

    alph = "NACGT"
    seq = []
    last = 47
    for o in outputs:
        #print(o.shape)
        am = np.argmax(o, axis=1)
        for x in am:
            #print(prd)
            #x = np.random.choice(5, p=prd)
            if x != 0 and x != last:
                seq.append(alph[x])
            last = x
    seq = "".join(seq)

    return seq


def call_file(filename):
    out = []
    with get_fast5_file(filename, mode="r") as f5:
        for read in f5.get_reads():
            read_id = read.get_read_id()
            signal = read.get_raw_data()
            signal = rescale_signal(signal)

            out.append((read_id, call_signal(signal)))
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
    for fn in files:
        start = datetime.datetime.now()
        for read_id, basecall in call_file(fn):
            print(">%s" % read_id, file=fout)
            print(basecall, file=fout)
            done += 1
            print("done %d/%d" % (done, len(files)), read_id, datetime.datetime.now() - start)
