#!/usr/bin/env python

from ont_fast5_api.fast5_interface import get_fast5_file
import argparse
import os
import numpy as np
import datetime
import deepnano2
from torch import multiprocessing as mp
from torch.multiprocessing import Pool
import torch
from deepnano2.gpu_model import Net
from tqdm import tqdm

# TODO: change cuda:0 into something better

step = 550
pad = 25
reads_in_group = 100
torch.set_grad_enabled(False)

def caller(model, qin, qout):
    while True:
        item = qin.get()
        if item is None:
            qout.put(None)
            return 
        read_marks, batch = item
        net_result = model(batch)
        qout.put((read_marks, net_result))

def finalizer(fn, qin):
    fo = open(fn, "w")
    alph = np.array(["N", "A", "C", "G", "T"])
    cur_rows = []
    while True:
        item = qin.get()
        if item is None:
            if len(cur_rows) > 0:
                stack = np.vstack(cur_rows)
                seq = deepnano2.beam_search_py(stack, 5, 0.1)
                print(seq, file=fo)
            return
        read_marks, res = item
        res = res.to(device='cpu', dtype=torch.float32).numpy()
        for read_mark, row in zip(read_marks, res):
            if read_mark is not None:
                if len(cur_rows) > 0:
                    stack = np.vstack(cur_rows)
                    seq = deepnano2.beam_search_py(stack, 5, 0.1)
                    print(seq, file=fo)
                    cur_rows = []
                print(">%s" % read_mark, file=fo)
            cur_rows.append(row[pad:-pad])


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

if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='Fast caller for ONT reads')

    parser.add_argument('--directory', type=str, nargs='*', help='One or more directories with reads')
    parser.add_argument('--reads', type=str, nargs='*', help='One or more read files')
    parser.add_argument("--output", type=str, required=True, help="Output FASTA file name")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads for basecalling")
    parser.add_argument("--weights", type=str, default=None, help="Path to network weights")
    parser.add_argument("--network-type", choices=["fast", "accurate"], default="fast")
    parser.add_argument("--beam-size", type=int, default=None, help="Beam size (defaults 5 for fast and 20 for accurate. Use 1 to disable.")
    parser.add_argument("--beam-cut-threshold", type=float, default=None, help="Threshold for creating beams (higher means faster beam search, but smaller accuracy)")
    parser.add_argument('--half', dest='half', action='store_true', help='Use half precision (fp16) during basecalling. On new graphics card, it might speed things up')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for calling, longer ones are usually faster, unless you get GPU OOM error')
    parser.set_defaults(half=False)

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
    if args.half:
        model.half()

    model.share_memory()

    qcaller = mp.Queue(10)
    qfinalizer = mp.Queue()
    call_proc = mp.Process(target=caller, args=(model, qcaller, qfinalizer))
    final_proc = mp.Process(target=finalizer, args=(args.output, qfinalizer))
    call_proc.start()
    final_proc.start()

    chunk_dtype = torch.float16 if args.half else torch.float32

    start_time = datetime.datetime.now()
    print("start", start_time)
    chunks = []
    read_marks = []
    for fn in tqdm(files):
        try:
            with get_fast5_file(fn, mode="r") as f5:
                for read in f5.get_reads():
                    read_id = read.get_read_id()
                    signal = rescale_signal(read.get_raw_data())
                    for i in range(0, len(signal), 3*step):
                        if i + 3*step + 6*pad > len(signal):
                            break
                        part = np.array(signal[i:i+3*step+6*pad])    
                        if i == 0:
                            read_marks.append(read_id)
                        else:
                            read_marks.append(None)
                        chunks.append(np.vstack([part, part * part]).T)
                        if len(chunks) == args.batch_size:
                            qcaller.put((read_marks, torch.tensor(np.stack(chunks), dtype=chunk_dtype, device='cuda:0')))
                            chunks = []
                            read_marks = []
        except OSError:
            # TODO show something here
            pass
    if len(chunks) > 0:
        qcaller.put((read_marks, torch.tensor(np.stack(chunks), dtype=chunk_dtype, device='cuda:0')))

    qcaller.put(None)
    call_proc.join()
    final_proc.join()
    print("fin", datetime.datetime.now() - start_time)

