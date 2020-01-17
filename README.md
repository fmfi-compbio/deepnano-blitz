# Ultra fast ONT basecaller

This is a very fast basecaseller which can basecall reads as fast as they come
from MinION.

## Limitations

* Works only on 64bit linux.
* On AMD cpus it is advised to use: `export MKL_DEBUG_CPU_TYPE=5`

## Instalation

* Clone this repository
* Run `python setup.py`

## Running

`deepnano2_caller.py --output out.fasta --directory reads_directory/`
