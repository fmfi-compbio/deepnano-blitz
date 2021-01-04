# Ultra fast ONT basecaller

This is a very fast basecaseller which can basecall reads as fast as they come
from MinION on ordinary laptop.

If you find this work useful, please cite (there will be preprint about his updates coming soon):

[Vladimir Boza, Peter Peresini,  Brona Brejova,  Tomas Vinar. "DeepNano-blitz: A Fast Base Caller for MinION Nanopore Sequencers." bioRxiv 2020.02.11.944223; doi: https://doi.org/10.1101/2020.02.11.944223](https://www.biorxiv.org/content/10.1101/2020.02.11.944223v1)

## Limitations

* Tested only on 64bit linux (external parties made it work on MacOS, see below).
* Only R9.4.1 for now.
* On AMD CPUs it is advised to use: `export MKL_DEBUG_CPU_TYPE=5`
* You need python3 (tested with python 3.6). PyO3 package needs at least python3.5.
* In some situations you might need to do `export OMP_NUM_THREADS=1`

## Instalation

* Install Rust (programming language, not game and besides you should already have it ;) ). You can view instuctions here: https://www.rust-lang.org/tools/install and just run `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
* Ask for nightly version `rustup default nightly-2021-01-03` (other nightly versions might also work, but this one is tested).
* Prepare your conda environment (`conda create python=3.9 --name deepnanoblitz`). You do not have to do this, if you manage your Python packages in different way. And activate it (`conda activate deepnanoblitz`).
* Clone this repository (`git clone https://github.com/fmfi-compbio/deepnano-blitz.git`)
* Go inside the package (`cd deepnano-blitz`)
* Run `python setup.py install`
* Change Rust version to whatever you like (not needed)

## Instalation via binary wheel

* Prepare your environment (conda, virtualenv, ...) with python3.6.
* Warning outdated: On Linux machine with python3.6 (and CPU with AVX2 instructions) you can use: `pip install dist/deepnano2-0.1-cp36-cp36m-linux_x86_64.whl`


## Installing on Mac/Windows (not tested)

We heavily rely on MKL libraries (downloaded from here https://anaconda.org/intel/mkl-static/files via https://github.com/fmfi-compbio/deepnano-blitz/blob/master/build.rs#L17). Changing download URL (and maybe some filenames) should work. If yes, please send us pointer to tested configuration (we will then add platform detection and relevant code branches to master).

## Running

Try one off (ordered by increasing accuracy and decresing speed):

* `deepnano2_caller.py --output out.fasta --directory reads_directory/ --network-type 48 --beam-size 1`
* `deepnano2_caller.py --output out.fasta --directory reads_directory/ --network-type 48`
* `deepnano2_caller.py --output out.fasta --directory reads_directory/ --network-type 56`
* `deepnano2_caller.py --output out.fasta --directory reads_directory/ --network-type 64`
* `deepnano2_caller.py --output out.fasta --directory reads_directory/ --network-type 80`
* `deepnano2_caller.py --output out.fasta --directory reads_directory/ --network-type 96`

You can also increase number of threads:

`deepnano2_caller.py --output out.fasta --directory reads_directory/ --threads 4`

Or ask for fastq output or compressed output:
`deepnano2_caller.py --output out.fastq --directory reads_directory/ --output-format fastq`
`deepnano2_caller.py --output out.fasta.gz --directory reads_directory/ --gzip-output`

For more accurate (but much slower) basecalling run:
`deepnano2_caller.py --output out.fasta --directory reads_directory/ --threads 16 --network-type 256`

You can check the installion via:
`deepnano2_caller.py --output testx.fastq --reads test_sample/*.fast5 --network-type 64 --beam-size 5 --threads 1 --output-format fastq`

And compare it to `test_sample/expected.fastq`

## Calling programmatically

```python
import deepnano2
import os
import numpy as np

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

network_type = "48"
beam_size = 5
beam_cut_threshold = 0.01
weights = os.path.join(deepnano2.__path__[0], "weights", "rnn%s.txt" % network_type)
caller = deepnano2.Caller(network_type, weights, beam_size, beam_cut_threshold)

# Minimal size for calling is STEP*3 + PAD*6 + 1 (STEP and PAD are defined in src/lib.rs)
signal = np.random.normal(size=(1000*3+10*6+1))

signal = rescale_signal(signal)

print(caller.call_raw_signal(signal))
```

When using only very short first part of read (and especially when you lowered the STEP), you might
need to cutoff bad part of the read (usually called the stall) using this (trim function is adopted from nanonet):

```python
def trim(signal, window_size=40, threshold_factor=3.0, min_elements=3):

    med, mad = med_mad(signal[-(window_size*25):])
    threshold = med + mad * threshold_factor
    num_windows = len(signal) // window_size

    for pos in range(num_windows):

        start = pos * window_size
        end = start + window_size

        window = signal[start:end]

        if len(window[window > threshold]) > min_elements:
            if window[-1] > threshold:
                continue
            return end

    return 0

# Call this before normalization
start = trim(signal)
signal = signal[start:]
```

Or alternativelly you can use this triming function (courtesy of https://github.com/Gogis0):

```python
def trim_blank(sig, window=300):
    N = len(sig)
    variances = [np.var(sig[i:i+window]) for i in range(N//2, N-window, window)]
    mean_var = np.mean(variances)
    trim_idx = 20
    while window > 5:
        while np.var(sig[trim_idx: trim_idx + window]) < 0.3*mean_var:
            trim_idx += 1
        window //= 2

    return trim_idx
```


## Benchmarks

Run on subset of reads from [Klebsiela
dataset](https://github.com/rrwick/Basecalling-comparison/tree/95bf07476f61cda79e6971f20f48c6ac83e634b3)
and also on human dataset.

MinION theoretical maximum is 2M signals/s (which in reality never happens, so realistic number is
around 1.5 M signals/s).

### Basecallers in fast mode

Speed is given in signals/sec.

| Network type        | Laptop 1 core | Laptop 4 cores | Xeon 1 core | Xeon 4 cores | Klebs Mapped % | Klebsiela 10% acc | Klebs median acc | Klebs 90% acc | Human mapped % | Human median acc | Human 90% acc |
|---------------------|---------------|----------------|-------------|--------------|----------------|-------------------|----------------------|-------------------|---------------|------------------|---------------|
| w48, beam1          | 1.4M          | 4.4M           | 1.0M        | 4.1M         | 98.6           | 73.0              | 84.0                 | 88.8              | 84.3           | 80.6             | 86.8          |
| w48, beam5 | 1.3M          | 3.8M           | 920.6K      | 3.8M         | 98.8           | 75.2              | 85.1                 | 89.5              | 84.9           | 81.6             | 87.7          |
| w56, beam5 | 941.8K        | 2.8M           | 649.7K      | 2.6M         | 98.9           | 76.1              | 85.9                 | 90.2              | 86.1           | 82.3             | 88.5          |
| w64, beam5 | 728.8K        | 2.1M           | 486.3K      | 2.1M         | 99.0           | 77.2              | 86.6                 | 90.8              | 85.5           | 83.4             | 89.3          |
| w80, beam5 | 477.6K        | 1.5M           | 351.5K      | 1.4M         | 99.0           | 77.3              | 87.3                 | 91.6              | 86.1           | 84.3             | 89.8          |
| w96, beam5 | 324.1K        | 1.0M           | 249.1K      | 1.0M         | 99.3           | 79.0              | 88.4                 | 92.4              | 87.4           | 85.9             | 91.0          |
| guppy 3.4.4 fast    | 87.9K         | 328.6K         | 66.4K       | 264.4K       | 99.5           | 79.6              | 88.4                 | 92.5              | 89.1           | 85.1             | 91.0          |
| guppy 3.4.4 hac     | 9.5K          | 35.1K          | 7.2K        | 29.1K        | 99.5           | 81.6              | 90.6                 | 94.5              | 89.6           | 87.4             | 93.3          |

## GPU version (experimental, not advised to use)

There is also GPU version of basecaller, which is slightly worse and slower than guppy,
but **does not** require compute capability 6.2 (anything which can run Pytorch 1.0 is good enough).

If you want to use GPU version, we recommend setting up Conda environment with pytorch first.


It can be run like this:
`deepnano2_caller_gpu.py --output out.fasta --directory reads_directory/`

If you have new GPU (like RTX series) this might be faster:
`deepnano2_caller_gpu.py --output out.fasta --directory reads_directory/ --half --batch-size 2048`

### Basecallers in high-accuracy mode

Note, that we are using 16 threads. Results on klebsiella dataset.

| Basecaller                                       | Time to basecall | 10%-percentile accuracy | Median accuracy | 90%-percentile accuracy |
|--------------------------------------------------|             ----:|                --------:|            ----:|                 -------:|
| Guppy 3.3.0 hac, 16 threads XEON E5-2695 v4      | 18m 16s          | 82.5%                   | 89.8%           | 93.8%                   |
| DN-blitz big, 16 threads XEON E5-2695 v4         | 12m 5s           | 82.1%                   | 89.1%           | 93.2%                   |
| DN-blitz big, 16 threads XEON E5-2695 v4, beam   | 12m 30s          | 83.4%                   | 89.8%           | 93.6%                   |
|--------------------------------------------------|              ----|                 --------|             ----|                  -------|
| Guppy 3.3.0 hac, Titan XP GPU                    | 36s              | 82.5%                   | 89.8%           | 93.8%                   |
| DN-blitz big, Titan XP GPU                       | 54s              | 82.2%                   | 89.2%           | 93.2%                   |
|--------------------------------------------------|              ----|                 --------|             ----|                 -------|
| DN-blitz big, Quadro M1200 GPU (laptop)          | 4m 5s            | 82.2%                   | 89.2%           | 93.2%                   |
| Guppy 3.3.0 hac,  Quadro M1200 GPU (laptop)      | N/A              | N/A                     | N/A             | N/A                     |

TODO: beam search for GPU version and RTX results

