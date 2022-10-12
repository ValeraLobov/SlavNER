from setuptools import find_packages, setup

setup(
    name="huawei-slavic-ner-models",
    version="0.0.1",
    packages=find_packages(),
    package_data={
        '': ['*.yaml'],
    },
    entry_points={
        "console_scripts": ["huawei-slavic-ner-models = huawei_slavic_ner_models.worker:main"]
    },
)
