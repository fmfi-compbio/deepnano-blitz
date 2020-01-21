# Ultra fast ONT basecaller

This is a very fast basecaseller which can basecall reads as fast as they come
from MinION on ordinary laptop.
There is also GPU version of basecaller, which (soon) will be as good as Guppy
and **does not** require compute capability 6.2 (anything which can run Pytorch 1.0 is good enough).

Also contains bigger network, which has performance similar to Guppy.

If you find this work useful, please cite:

[Boža, Vladimír, Broňa Brejová, and Tomáš Vinař. "DeepNano: deep recurrent neural networks for base calling in MinION nanopore reads." PloS one 12.6 (2017).](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0178751)

## Limitations

* Works only on 64bit linux.
* Only R9.4.1 for now.
* On AMD CPUs it is advised to use: `export MKL_DEBUG_CPU_TYPE=5`
* You need python3 (tested with python 3.6)

## Instalation

* [optional] If you want to use GPU, we recommend setting up Conda environment with pytorch first.
* Install Rust (programming language, not game and besides you should already have it ;) )
* Ask for nightly version `rustup default nightly-2019-12-11`
* Clone this repository
* Run `python setup.py`
* Change Rust version to whatever you like

## Running

`deepnano2_caller.py --output out.fasta --directory reads_directory/`

For more accurate (but much slower) basecalling run:
`deepnano2_caller.py --output out.fasta --directory reads_directory/ --threads 16 --network-type accurate`

GPU caller:
`deepnano2_caller_gpu.py --output out.fasta --directory reads_directory/`

If you have new GPU (like RTX series) this might be faster:
`deepnano2_caller_gpu.py --output out.fasta --directory reads_directory/ --half --batch-size 2048`

## Benchmarks

Run on subset of 476 reads from [Klebsiela dataset](https://github.com/rrwick/Basecalling-comparison/tree/95bf07476f61cda79e6971f20f48c6ac83e634b3).
MinION produces this amount in apx. 52 seconds assuming maximum throughput of 2M signals/s (which in reality never
happens, so realistic number is around 70 seconds).

### Basecallers in fast mode

| Basecaller                                       | Time to basecall | Signals/s | 10%-percentile accuracy | Median accuracy | 90%-percentile accuracy |
|--------------------------------------------------|             ----:|----------:|                --------:|            ----:|                 -------:|
| Guppy 3.3.0 fast, 1 thread XEON E5-2695 v4       | 26m 0s           |    67,6k  | 80.4%                   | 87.6%           | 91.8%                   |
| Guppy 3.3.0 fast, 4 threads XEON E5-2695 v4      | 6m 54s           |    254k   | 80.4%                   | 87.6%           | 91.8%                   |
| DN-blitz, 1 thread XEON E5-2695 v4               | 2m 3s            |    857k   | 75.5%                   | 83.9%           | 88.7%                   |
| DN-blitz, 4 threads XEON E5-2695 v4              | 32s              |    3.30M  | 75.5%                   | 83.9%           | 88.7%                   |
| DN-blitz, 1 thread XEON E5-2695 v4, beam         | 2m 12s           |    799k   | 77.4%                   | 85.1%           | 89.2%                   |
| DN-blitz, 4 threads XEON E5-2695 v4, beam        | 34s              |    3.10M  | 77.4%                   | 85.1%           | 89.2%                   |
|------------------------------------------        |             -----|-----------|                ---------|            -----|                 --------|
| DN-blitz, 1 thread i7-7700HQ (laptop)            | 1m 24s           |    1.25M  | 75.5%                   | 84.0%           | 88.7%                   |
| DN-blitz, 4 threads i7-7700HQ (laptop)           | 28s              |    3.77M  | 75.5%                   | 84.0%           | 88.7%                   |
| DN-blitz, 1 thread i7-7700HQ (laptop), beam      | 1m 33s           |    1.13M  | 77.5%                   | 85.1%           | 89.3%                   |
| DN-blitz, 4 threads i7-7700HQ (laptop), beam     | 29s              |    3.64M  | 77.5%                   | 85.1%           | 89.3%                   |

### Basecallers in high-accuracy mode

Note, that we are using 16 threads.

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

### Accuracy on Human data

(TODO exact source)

| Basecaller                                       | 10%-percentile accuracy | Median accuracy | 90%-percentile accuracy |
|--------------------------------------------------|                --------:|            ----:|                 -------:|
| Guppy 3.3.0 hac,                                 | 57.7%                   | 88.2%           | 93.4%                   |
| Guppy 3.3.0 fast                                 | 58.9%                   | 85.7%           | 91.0%                   |
| DN-blitz                                         | 57.3%                   | 81.1%           | 86.8%                   |
| DN-blitz, beam                                   | 55.2%                   | 81.6%           | 87.5%                   |
| DN-blitz big                                     | 56.5%                   | 86.7%           | 91.9%                   |
| DN-blitz big, beam                               | 59.2%                   | 87.6%           | 92.4%                   |
