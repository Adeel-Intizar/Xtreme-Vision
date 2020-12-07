from setuptools import setup, find_packages
from xtreme_vision import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
      name = 'xtreme_vision',
      version = __version__,
      description = 'A Python Library for Computer Vision tasks like Object Detection, Segmentation, Pose Estimation etc',
      url = "https://github.com/Adeel-Intizar/Xtreme-Vision",
      author = "Adeel Intizar",
      author_email = "kingadeel2017@outlook.com",
      maintainer = "Adeel Intizar",
      maintainer_email = "kingadeel2017@outlook.com",
      long_description = long_description,
      long_description_content_type="text/markdown",
      packages = find_packages(),
      python_requires='>=3.5, <4',
      classifiers = [
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Information Technology',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3 :: Only',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: Implementation',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Image Processing',
          'Topic :: Scientific/Engineering :: Image Recognition',
          'Topic :: Software Development :: Libraries :: Python Modules'
          ],
      keywords = [
      "object detection", 
      "computer vision",
      "pose estimation", 
      'machine learning', 
      'deep learning', 
      'artificial intelligence',
      'xtreme vision', 
      'segmentation',
      'yolo', 
      'retinanet',
      'centernet', 
      'yolov4', 
      'deeplab',
      'tinyyolo'],
      
      install_requires = [
          "tensorflow", 
          'keras',
          'opencv-python',
          'numpy',
          'pillow',
          'matplotlib',
          'pandas',
          'scikit-learn',
          'progressbar2',
          'scipy',
          'h5py',
	  'imgaug',
	  'scikit-image',
	  'labelme2coco',
	  'configobj',
	  ],
      
      project_urls={
          
        'Bug Reports': 'https://github.com/Adeel-Intizar/Xtreme-Vision/issues',
        'Funding': 'https://patreon.com/adeelintizar',
        'Say Thanks!': 'https://saythanks.io/to/kingadeel2017%40outlook.com',
        'Source': 'https://github.com/Adeel-Intizar/Xtreme-Vision/'},
      
      )