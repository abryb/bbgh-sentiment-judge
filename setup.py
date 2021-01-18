import setuptools

with open('requirements.txt') as f:
    reqs = f.read().splitlines()

setuptools.setup(
    name="trainer",
    version="0.0.1",
    author="Błażej Rybarkiewicz",
    author_email="b.rybarkiewicz@gmail.com",
    description="Sentiment trainer module for Piłkomentarz project",
    url="https://github.com/abryb/bbgh-sentiment-judge",
    packages=setuptools.find_packages(),
    install_requires=reqs,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
