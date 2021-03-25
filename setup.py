#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import glob
import os
import shutil
from os import path
from setuptools import find_packages, setup
from typing import List
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension
from torch.utils.hipify import hipify_python

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 5], "Requires PyTorch >= 1.5"


def get_version():
    init_py_path = path.join(path.abspath(path.dirname(__file__)), "detectron2", "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")

    # The following is used to build release packages.
    # Users should never use it.
    suffix = os.getenv("D2_VERSION_SUFFIX", "")
    version = version + suffix
    if os.getenv("BUILD_NIGHTLY", "0") == "1":
        from datetime import datetime

        date_str = datetime.today().strftime("%y%m%d")
        version = version + ".dev" + date_str

        new_init_py = [l for l in init_py if not l.startswith("__version__")]
        new_init_py.append('__version__ = "{}"\n'.format(version))
        with open(init_py_path, "w") as f:
            f.write("".join(new_init_py))
    return version


def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "detectron2", "layers", "csrc")

    main_source = path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))

    from torch.utils.cpp_extension import ROCM_HOME

    is_rocm_pytorch = (
        True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False
    )

    hipify_ver = (
        [int(x) for x in torch.utils.hipify.__version__.split(".")]
        if hasattr(torch.utils.hipify, "__version__")
        else [0, 0, 0]
    )

    if is_rocm_pytorch and hipify_ver < [1, 0, 0]:

        # Earlier versions of hipification and extension modules were not
        # transparent, i.e. would require an explicit call to hipify, and the
        # hipification would introduce "hip" subdirectories, possibly changing
        # the relationship between source and header files.
        # This path is maintained for backwards compatibility.

        hipify_python.hipify(
            project_directory=this_dir,
            output_directory=this_dir,
            includes="/detectron2/layers/csrc/*",
            show_detailed=True,
            is_pytorch_extension=True,
        )

        source_cuda = glob.glob(path.join(extensions_dir, "**", "hip", "*.hip")) + glob.glob(
            path.join(extensions_dir, "hip", "*.hip")
        )

        shutil.copy(
            "detectron2/layers/csrc/box_iou_rotated/box_iou_rotated_utils.h",
            "detectron2/layers/csrc/box_iou_rotated/hip/box_iou_rotated_utils.h",
        )
        shutil.copy(
            "detectron2/layers/csrc/deformable/deform_conv.h",
            "detectron2/layers/csrc/deformable/hip/deform_conv.h",
        )

        sources = [main_source] + sources
        sources = [
            s
            for s in sources
            if not is_rocm_pytorch or torch_ver < [1, 7] or not s.endswith("hip/vision.cpp")
        ]

    else:

        # common code between cuda and rocm platforms,
        # for hipify version [1,0,0] and later.

        source_cuda = glob.glob(path.join(extensions_dir, "**", "*.cu")) + glob.glob(
            path.join(extensions_dir, "*.cu")
        )

        sources = [main_source] + sources

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and ((CUDA_HOME is not None) or is_rocm_pytorch)) or os.getenv(
        "FORCE_CUDA", "0"
    ) == "1":
        extension = CUDAExtension
        sources += source_cuda

        if not is_rocm_pytorch:
            define_macros += [("WITH_CUDA", None)]
            extra_compile_args["nvcc"] = [
                "-O3",
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ]
        else:
            define_macros += [("WITH_HIP", None)]
            extra_compile_args["nvcc"] = []

        if torch_ver < [1, 7]:
            # supported by https://github.com/pytorch/pytorch/pull/43931
            CC = os.environ.get("CC", None)
            if CC is not None:
                extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "detectron2._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


def get_model_zoo_configs() -> List[str]:
    """
    Return a list of configs to include in package for model zoo. Copy over these configs inside
    detectron2/model_zoo.
    """

    # Use absolute paths while symlinking.
    source_configs_dir = path.join(path.dirname(path.realpath(__file__)), "configs")
    destination = path.join(
        path.dirname(path.realpath(__file__)), "detectron2", "model_zoo", "configs"
    )
    # Symlink the config directory inside package to have a cleaner pip install.

    # Remove stale symlink/directory from a previous build.
    if path.exists(source_configs_dir):
        if path.islink(destination):
            os.unlink(destination)
        elif path.isdir(destination):
            shutil.rmtree(destination)

    if not path.exists(destination):
        try:
            os.symlink(source_configs_dir, destination)
        except OSError:
            # Fall back to copying if symlink fails: ex. on Windows.
            shutil.copytree(source_configs_dir, destination)

    config_paths = glob.glob("configs/**/*.yaml", recursive=True)
    return config_paths


# For projects that are relative small and provide features that are very close
# to detectron2's core functionalities, we install them under detectron2.projects
PROJECTS = {
    "detectron2.projects.point_rend": "projects/PointRend/point_rend",
    "detectron2.projects.deeplab": "projects/DeepLab/deeplab",
    "detectron2.projects.panoptic_deeplab": "projects/Panoptic-DeepLab/panoptic_deeplab",
}

setup(
    name="detectron2",
    version=get_version(),
    author="FAIR",
    url="https://github.com/facebookresearch/detectron2",
    description="Detectron2 is FAIR's next-generation research "
    "platform for object detection and segmentation.",
    packages=find_packages(exclude=("configs", "tests*")) + list(PROJECTS.keys()),
    package_dir=PROJECTS,
    package_data={"detectron2.model_zoo": get_model_zoo_configs()},
    python_requires=">=3.6",
    install_requires=[
        # Do not add opencv here. Just like pytorch, user should install
        # opencv themselves, preferrably by OS's package manager, or by
        # choosing the proper pypi package name at https://github.com/skvark/opencv-python
        "termcolor>=1.1",
        "Pillow>=7.1",  # or use pillow-simd for better performance
        "yacs>=0.1.6",
        "tabulate",
        "cloudpickle",
        "matplotlib",
        "tqdm>4.29.0",
        "tensorboard",


        # # i.21.3.24.11:43) 갑자기 어젯밤부터 트레이닝이 안되는데, 트레이닝 시작할 웨잇 불러오는과정에서 
        # #  fvcore 버전이랑 iopath 버전 호환이 안되면 문제가 발생하는듯해서 fvcore랑 iopath 버전히스토리 확인해보니, 
        # #  둘다 아주 최근 뭐 거의 몇시간전, 바로어제 새버전 릴리즈되고 이런걸로봐서 여기서 문제가 있는듯해서,
        # #  Det2 리포지토리 가서 setup.py 보니, 역시 뭔가가 바뀌어있음. 
        # #    일단 논문2(치과파노라마에서 panoptic seg 하는거. panoptic deeplab 이용.) 끝날때까지는, 내가 Git 에대해 아직 잘 몰라서 혹시 문제생길지 모르니
        # #  Det2_mvdeltGithub 을 Det2 원 리포지토리의 업데이트를 반영해서 업데이트(git fetch & merge 등) 하지않으려해서,
        # #  일단 내가 직접 이렇게 수정된부분 복붙해놓음.
        # #
        # # Lock version of fvcore/iopath because they may have breaking changes
        # # NOTE: when updating fvcore/iopath version, make sure fvcore depends
        # # on the same version of iopath.
        # "fvcore>=0.1.4,<0.1.5",  # required like this to make it pip installable
        # "iopath>=0.1.7,<0.1.8",
        #
        # # i.21.3.25.10:09) ->위처럼 해도 안됨. 이 버전들은 Det2 최신버전(현재 0.4릴리즈됏음. 난 그전에 포크한거 내가수정해서 쓰고있는중인거고.)
        # #  일때는 아마 잘 될텐데, 지금 내 Det2 버전상태에선 잘 안되나봄. 걍 며칠전까지 잘되던 fvcore,iopath버전으로 해줘야할듯.
        "fvcore>=0.1.3,<0.1.4",  # required like this to make it pip installable <- fvcore 는 기존코드와 동일.
        "iopath>=0.1.2,<0.1.7", # <- iopath만 버전 상한선(<0.1.7) 설정만 해줬음. 기존코드에선 상한선이 없었음.
        

        # i.21.3.24.11:50) 요 두줄이 기존 코드.
        # "fvcore>=0.1.3,<0.1.4",  # required like this to make it pip installable
        # "iopath>=0.1.2",


        "pycocotools>=2.0.2",  # corresponds to https://github.com/ppwwyyxx/cocoapi
        "future",  # used by caffe2
        "pydot",  # used to save caffe2 SVGs
        

        # i.21.3.25.0:46) fvcore 랑 iopath Det2최신깃헙소스대로 바꿧는데도 안돼서, 혹시
        #  얘네들도 추가됏길래 얘네도 설치해주면 되려나 해서 붙여놔봄.(기존엔 이 두개는 없었음.)
        #    ->자야하는데 Det2 다시설치할 시간 없어서 
        #      걍 fvcore 랑 iopath 를 예전버전으로 pip 으로 설치해주니 일단 되네(코랩에서). /21.3.25.1:00.
        #      # i.21.3.25.10:08) 자고나서 보니 체크포인트 저장할때 iopath 관련 에러낫네;;; 
        #         다시돌려볼시간없음. 걍 며칠전까지 잘되던 딱 그버전으로 해줘야할듯.
        # "dataclasses; python_version<'3.7'",
        # "omegaconf==2.1.0.dev22",

    ],
    extras_require={
        "all": [
            "shapely",
            "psutil",
            "hydra-core",
            "panopticapi @ https://github.com/cocodataset/panopticapi/archive/master.zip",
        ],
        "dev": [
            "flake8==3.8.1",
            "isort==4.3.21",
            "black==20.8b1",
            "flake8-bugbear",
            "flake8-comprehensions",
        ],
    },
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
