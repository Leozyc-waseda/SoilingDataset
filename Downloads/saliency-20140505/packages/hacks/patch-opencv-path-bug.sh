#!/bin/sh

# although opencv includes are in /usr/include/opencv, they include
# each other without specifying the 'opencv' stem, hence assuming that
# /usr/include/opencv is in the include path. This is dirty! And this
# patch fixes this.

R='./packages/hacks/replace'


for x in `/bin/ls /usr/include/opencv`; do
    $R "#include <$x>" "#include <opencv/$x>" -- /usr/include/opencv/*
    $R "#include \"$x\"" "#include <opencv/$x>" -- /usr/include/opencv/*
done
