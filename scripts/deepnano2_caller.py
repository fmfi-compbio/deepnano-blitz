#!/usr/bin/env python

from ont_fast5_api.fast5_interface import get_fast5_file
import argparse
import os
import numpy as np
import datetime
import deepnano2
from multiprocessing import Pool

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
    return signal

def call_file(filename):
    out = []
    try:
        with get_fast5_file(filename, mode="r") as f5:
            for read in f5.get_reads():
                read_id = read.get_read_id()
                signal = read.get_raw_data()
                signal = rescale_signal(signal)

                out.append((read_id, caller.call_raw_signal(signal)))
    except OSError:
        return []
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fast caller for ONT reads')

    parser.add_argument('--directory', type=str, nargs='*', help='One or more directories with reads')
    parser.add_argument('--reads', type=str, nargs='*', help='One or more read files')
    parser.add_argument("--output", type=str, required=True, help="Output FASTA file name")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads for basecalling")
    parser.add_argument("--weights", type=str, default=None, help="Path to network weights")
    parser.add_argument("--network-type", choices=["fast", "accurate"], default="fast")
    parser.add_argument("--beam-size", type=int, default=None,
        help="Beam size (defaults 5 for fast and 20 for accurate). Use 1 to disable.")
    parser.add_argument("--beam-cut-threshold", type=float, default=None,
        help="Threshold for creating beams (higher means faster beam search, but smaller accuracy). Values higher than 0.2 might lead to weird errors.")

    args = parser.parse_args()

    assert args.threads >= 1

    files = args.reads if args.reads else []
    if args.directory:
        for directory_name in args.directory:
            files += [os.path.join(directory_name, fn) for fn in os.listdir(directory_name)]

    if len(files) == 0:
        print("Zero input reads, nothing to do.")
        sys.exit()

    if args.weights is None:
        if args.network_type == "accurate":
            weights = os.path.join(deepnano2.__path__[0], "weights", "weightsbig520.txt")
        else:
            weights = os.path.join(deepnano2.__path__[0], "weights", "weightsn460.txt")
    else:
        weights = args.weights

    if args.beam_size is None:
        beam_size = 5 if args.network_type == "fast" else 20
    else:
        beam_size = args.beam_size

    if args.beam_cut_threshold is None:
        beam_cut_threshold = 0.1 if args.network_type == "fast" else 0.0001
    else:
        beam_cut_threshold = args.beam_cut_threshold

    caller = deepnano2.Caller(args.network_type, weights, beam_size, beam_cut_threshold)


    fout = open(args.output, "w")

    if args.threads <= 1:
        done = 0
        for fn in files:
            start = datetime.datetime.now()
            for read_id, basecall in call_file(fn):
                print(">%s" % read_id, file=fout)
                print(basecall, file=fout)
                done += 1
                print("done %d/%d" % (done, len(files)), read_id, datetime.datetime.now() - start)

    else:
        pool = Pool(args.threads)
        done = 0
        for out in pool.imap_unordered(call_file, files):
            for read_id, basecall in out:
                print(">%s" % read_id, file=fout)
                print(basecall, file=fout)
                done += 1
                print("done %d/%d" % (done, len(files)), read_id)
    
    fout.close()

