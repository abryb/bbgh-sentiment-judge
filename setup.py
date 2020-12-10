import setuptools

REQUIRED_PACKAGES = [
    'requests',
    'python-dotenv',
    'Unidecode',
    'setuptools',
    'pandas',
    'matplotlib',
    'livelossplot',
    'numpy',
    'tensorflow',
    'sklearn',
    'keras',
    'gensim',
    'h5py',
    'docopt'
]

setuptools.setup(
    name="trainer",
    version="0.0.1",
    author="Błażej Rybarkiewicz",
    author_email="b.rybarkiewicz@gmail.com",
    description="Sentiment judge module for Piłkomentarz project",
    url="https://github.com/abryb/bbgh-sentiment-judge",
    packages=setuptools.find_packages(),
    install_requires=REQUIRED_PACKAGES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)