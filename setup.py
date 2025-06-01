from setuptools import setup


setup(name="dosrixs",
      version="0.0.1",
      author="Harrison LaBollita",
      author_email="hlabollita@flatironinstitute.org",
      description="a library for computing RIXS and XAS spectrum from orbitally-resolved density of states",
      url="https://github.com/harrisonlabollita/dosrixs",
      packages=['dosrixs'],
      license='MIT',
      install_requires=["numpy", "scipy", "sympy"]
      )
