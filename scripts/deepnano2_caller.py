#!/usr/bin/env python

from ont_fast5_api.fast5_interface import get_fast5_file
import argparse
import os
import numpy as np
import datetime
import deepnano2
from multiprocessing import Pool
import sys
import gzip

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

def write_output(read_id, basecall, output_file, format):
    if format == "fasta":
        print(">%s" % read_id, file=fout)
        print(basecall, file=fout)
    else: # fastq
        print("@%s" % read_id, file=fout)
        print(basecall, file=fout)
        print("+", file=fout)
        print("".join("A" for _ in basecall), file=fout)
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fast caller for ONT reads')

    parser.add_argument('--directory', type=str, nargs='*', help='One or more directories with reads')
    parser.add_argument('--reads', type=str, nargs='*', help='One or more read files')
    parser.add_argument("--output", type=str, required=True, help="Output FASTA file name")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads for basecalling, default 1")
    parser.add_argument("--weights", type=str, default=None, help="Path to network weights, only used for custom weights")
    parser.add_argument("--network-type", choices=["48", "56", "64", "80", "96", "256"], default="48", help="Size of network. Default 48")
    parser.add_argument("--beam-size", type=int, default=None,
        help="Beam size (defaults 5 for 48,56,64,80,96 and 20 for 256). Use 1 for greedy decoding.")
    parser.add_argument("--beam-cut-threshold", type=float, default=None,
        help="Threshold for creating beams (higher means faster beam search, but smaller accuracy). Values higher than 0.2 might lead to weird errors. Default 0.1 for 48,...,96 and 0.0001 for 256")
    parser.add_argument("--output-format", choices=["fasta", "fastq"], default="fasta")
    parser.add_argument("--gzip-output", action="store_true", help="Compress output with gzip")

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
        weights = os.path.join(deepnano2.__path__[0], "weights", "rnn%s.txt" % args.network_type)
    else:
        weights = args.weights

    if args.beam_size is None:
        beam_size = 5 if args.network_type != "256" else 20
    else:
        beam_size = args.beam_size

    if args.beam_cut_threshold is None:
        beam_cut_threshold = 0.1 if args.network_type != "256" else 0.0001
    else:
        beam_cut_threshold = args.beam_cut_threshold

    caller = deepnano2.Caller(args.network_type, weights, beam_size, beam_cut_threshold)


    if args.gzip_output:
        fout = gzip.open(args.output, "wt")
    else:
        fout = open(args.output, "w")

    if args.threads <= 1:
        done = 0
        for fn in files:
            start = datetime.datetime.now()
            for read_id, basecall in call_file(fn):
                write_output(read_id, basecall, fout, args.output_format) 
                done += 1
                print("done %d/%d" % (done, len(files)), read_id, datetime.datetime.now() - start, file=sys.stderr)

    else:
        pool = Pool(args.threads)
        done = 0
        for out in pool.imap_unordered(call_file, files):
            for read_id, basecall in out:
                write_output(read_id, basecall, fout, args.output_format)
                done += 1
                print("done %d/%d" % (done, len(files)), read_id, file=sys.stderr)
    
    fout.close()

