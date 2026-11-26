"""Explicit numerical thread limits for repeatable reference experiments."""
from contextlib import contextmanager
import ipctk
from threadpoolctl import threadpool_limits,threadpool_info


@contextmanager
def numerical_threads(count=1):
    if count<1:raise ValueError('Thread count must be positive')
    previous=ipctk.get_num_threads()
    ipctk.set_num_threads(count)
    try:
        with threadpool_limits(limits=count):
            yield {'ipc_threads':ipctk.get_num_threads(),'blas':threadpool_info()}
    finally:
        ipctk.set_num_threads(previous)
