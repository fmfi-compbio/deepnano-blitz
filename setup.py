#!/usr/bin/env python
import sys

from setuptools import setup

try:
    from setuptools_rust import RustExtension, Binding
except ImportError:
    import subprocess

    errno = subprocess.call([sys.executable, "-m", "pip", "install", "setuptools-rust"])
    if errno:
        print("Please install setuptools-rust package")
        raise SystemExit(errno)
    else:
        from setuptools_rust import RustExtension, Binding

setup_requires = ["setuptools-rust>=0.10.1", "wheel"]
install_requires = ["ont-fast5-api>=2.0.1"]

setup(
    name="deepnano2",
    version="0.1",
    rust_extensions=[RustExtension("deepnano2.deepnano2", binding=Binding.PyO3, native=True, debug=False)],
    packages=["deepnano2"],
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False,
    install_requires=install_requires,
    setup_requires=setup_requires,
    scripts=["scripts/deepnano2_caller.py"]
)


