#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import torch

cxx_compiler_flags = []
extra_link_args = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")

# On Windows+HIP, c10.dll does not export the inherited c10::ValueError(SourceLocation,string)
# constructor (MSVC does not re-export inherited constructors). ext.cpp compiled by cl.exe
# picks up the dllimport declaration from ATen headers and LNK2001s. Redirect the missing
# thunk to c10::Error(SourceLocation,string) which IS exported; ValueError IS-A Error
# with no additional data members so the constructors are semantically identical.
if os.name == 'nt' and torch.version.hip:
    extra_link_args.append(
        "/ALTERNATENAME:__imp_??0ValueError@c10@@QEAA@USourceLocation@1@V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@Z"
        "=__imp_??0Error@c10@@QEAA@USourceLocation@1@V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@Z"
    )

setup(
    name="simple_knn",
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=[
            "spatial.cu", 
            "simple_knn.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": [], "cxx": cxx_compiler_flags},
            extra_link_args=extra_link_args)
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
