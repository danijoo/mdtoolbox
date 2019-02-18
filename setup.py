import setuptools

setuptools.setup(
    name = "mdtoolbox",
    version = "0.1",
    author = "Daniel Bauer",
    author_email = "mdtoolbox@headlezz.net",
    description = ("A set of functions used for analyzing MD trajectories"),
    license = "WTFPL",
    keywords = "MD biotite simulation",
    packages=["mdtoolbox"],
    install_requires=[
        'biotite',
        'numpy',
        'pandas',
        'matplotlib'
    ]
)

