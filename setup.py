from setuptools import setup, find_packages

# Setup file for steve1
setup(
    name='steve1',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'numpy>=1.19.0',
        'Pillow>=8.0.0',
        'pydantic>=2.0.0',
        'openai>=1.0.0',
        'python-dotenv>=1.0.0',
    ],
    extras_require={
        'llm': [
            'python-dotenv>=1.0.0',
            'openai>=1.0.0',
        ],
    },
)