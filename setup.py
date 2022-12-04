from setuptools import setup

dependencies = [
    "torch",
    "torch-geometric",
    # "torch-scatter",
    # "torch-cluster",
    # "torch-sparse",
    # "pyg-lib",
    "jupyter",
    "matplotlib",
    "notebook",
]

test_dependencies = [
    "torch",
    "pytest",
    "black",
    "isort",
    "tox",
    # "torch-geometric",
]

setup(
    name="GraphVision",
    version="0.0.1",
    packages=["src"],
    install_requires=dependencies,
    include_package_data=True,
    test_requires=test_dependencies,
    test_suite="tests",
)
