#!/usr/bin/env bash

# set the color variables of 'echo' cmd
DEF='\e[0m'
OK='\e[1;32m'

# update python packages
cd pkgs || exit
pip install . || exit

if [ "$1" = "-e" ];then
	pip install -e . || exit
fi

echo -e "${OK}Update OK!${DEF}"