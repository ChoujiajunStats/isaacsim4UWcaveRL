from setuptools import find_packages, setup


setup(
    name="isaac-underwater",
    version="0.1.0",
    description="Low-VRAM vectorized underwater navigation tasks for Isaac Lab",
    packages=find_packages(),
    install_requires=["pyyaml"],
    python_requires=">=3.11,<3.12",
    include_package_data=True,
    zip_safe=False,
)
