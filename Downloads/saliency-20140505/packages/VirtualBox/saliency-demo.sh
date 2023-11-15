#!/bin/sh

ezvision -K --movie --in=/home/mandriva/Videos/beverly03.mpg \
  --out=display --ior-type=None --rescale-input=320x240 \
  --nodisplay-interp-maps --maxnorm-type=FancyOne --display-foa
