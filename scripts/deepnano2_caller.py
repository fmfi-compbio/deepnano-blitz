#!/usr/bin/env python

from ont_fast5_api.fast5_interface import get_fast5_file
import argparse
import os
import numpy as np
import datetime
import deepnano2

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

parser = argparse.ArgumentParser(description='Fast caller for ONT reads')

parser.add_argument('--directory', type=str, nargs='*', help='One or more directories with reads')
parser.add_argument('--reads', type=str, nargs='*', help='One or more read files')
parser.add_argument("--output", type=str, required=True, help="Output FASTA file name")

args = parser.parse_args()

files = args.reads if args.reads else []
if args.directory:
    for directory_name in args.directory:
        files += [os.path.join(directory_name, fn) for fn in os.listdir(directory_name)]

if len(files) == 0:
    print("Zero input reads, nothing to do.")
    sys.exit()

fout = open(args.output, "w")

for fn in files:
    start = datetime.datetime.now()
    with get_fast5_file(fn, mode="r") as f5:
        for read in f5.get_reads():
            read_id = read.get_read_id()
            signal = read.get_raw_data()
            signal = rescale_signal(signal)

            t1 = datetime.datetime.now()
            base_call = deepnano2.call_raw_signal(signal)
            t2 = datetime.datetime.now()
            print(">%s" % read_id, file=fout)
            print(base_call, file=fout)
            fout.flush()
            print("done", read_id, t1 - start, t2 - start, datetime.datetime.now() - start)

fout.close()
