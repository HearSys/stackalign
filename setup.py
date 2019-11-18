from setuptools import setup

setup(name='stackalign',
      version='0.1',
      description='HörSys GmbH',
      url='',
      author='Samuel John',
      author_email='john.samuel@hoersys.de',
      license='MIT',
      packages=['stackalign'],
      install_requires=[
          'pattern_finder_gpu',
          'numpy',
          'scipy',
          'scikit-image',
          'tqdm',
          'pillow',
          'svg.path',
          'xattr'],
      zip_safe=False)
