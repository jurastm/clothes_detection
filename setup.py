from setuptools import setup, find_packages

VERSION = '0.1'

REQUIREMENTS = [
    'numpy>=1.17.1',
    'tensorflow>=2.2.0',
]

setup(
    name="clothes-detection",
    version=VERSION,
    author="Yura Vasiliuk",
    author_email="yura.vasiliuk@gmail.com",
    license="Proprietary",
    classifiers=[
        # 3 - Alpha
        # 4 - Beta
        # 5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    packages=['clothes_detection'],
    install_requires=REQUIREMENTS,
    package_data={
        'clothes_detection': ['graphs/*'],
    },
    entry_points={
        'console_scripts': [
            'clothes-detection=clothes_detection.clothes_detector:ClothesDetector'
        ]
    },
    url='https://github.com/jurastm/clothes_detection',
)