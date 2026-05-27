#!/usr/bin/env bash

since="$(date +%F) 06:30"

find . -maxdepth 1 -type f -newermt "$since" \
    -printf '%TY-%Tm-%Td %TH:%TM:%TS  %p\n' | sort
