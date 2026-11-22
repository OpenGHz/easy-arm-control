from setuptools import setup, find_packages

setup(
    name='airbot_play_control',
    version='1.1',
    packages=find_packages(),
    install_requires=['numpy','pyserial','robotics_tools'],
    author='GHz',
    author_email='ghz23@mails.tsinghua.edu.cn',
    description='AIRbotPlay Control Upper.',
    url='https://gitlab.com/OpenGHz/airbot_play_conrol_upper.git',
    license='MIT'
)