from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "gptzip._native",
            sources=[
                "gptzip/native/module.c",
                "gptzip/native/cdf.c",
                "gptzip/native/range.c",
                "gptzip/native/bits.c",
            ],
            extra_compile_args=[
                "-O3",
                "-std=c11",
                "-fno-fast-math",
                "-ffp-contract=off",
                "-Wall",
                "-Wextra",
                "-Werror=implicit-function-declaration",
            ],
        )
    ]
)
