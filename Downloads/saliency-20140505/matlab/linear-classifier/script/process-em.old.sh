#!/bin/sh

#nodes="i15 i15 i15 i15 i16 i16 i16 i16 i17 i17 localhost localhost localhost localhost i13 i13 i6 i6 i7"

nodes="n01 n01 n01 n01 n02 n02 n02 n02 n02 n03 n03 n03 n03 n04 n04 n04 n04 n05 n05 n05 n05 n07 n07 n07 n07 n08 n08 n08 n08 ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo"
#nodes="n01 n01 n01 n01 n02 n02 n02 n02 n02 n03 n03 n03 n03 n04 n04 n04 n04 n05 n05 n05 n05 n07 n07 n07 n07 n08 n08 n08 n08"
#nodes="ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo"

#beoqueue.pl -q -n "$nodes" /lab/tmpi1/u/rsvp/process_rsvp.pl /lab/tmpi1/u/rsvp/stim??_??? /lab/tmpi1/u/rsvp/stim_noTgt_???
#$HOME/rsvp/beoqueue.pl -q -n "$nodes" /lab/mundhenk/rsvp/process_rsvp.pl /lab/tmp/60/rsvp/stim??_???
#$HOME/rsvp/beoqueue.pl -q -n "$nodes" /lab/mundhenk/rsvp/process_rsvp.pl /lab/tmp/60/hard.post.mask/stim??_??? /lab/tmp/60/hard.pre.mask/stim??_??? /lab/tmp/60/hard.w.mask/stim??_???
$HOME/rsvp/beoqueue.pl -n "$nodes" /lab/mundhenk/linear-classifier/script/process_rsvp.pl /lab/raid/images/RSVP/fullSequence/stim??_???
#$HOME/rsvp/beoqueue.pl -n "$nodes" /lab/mundhenk/rsvp/process_rsvp.pl /lab/raid/images/RSVP/sequences-with-100-variants/stim??_???