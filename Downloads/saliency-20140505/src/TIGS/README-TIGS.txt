- /lab/raid/videos/eye-tracking/vgames/vgames2/
  - test/
    - jbauf = james bond agent under fire
    - nfsu = need for speed: underground
    - tg = top gun
    - jbauf1-andrea1/
      - jbauf1-andrea1.eyeS = calibrated eyetracking data
      - jbauf1-andrea1.eyeS.ntrash = # of eye samples to skip at start
      - jbauf1-andrea1.eyeS.ppd = pixels-per-degree
      - jbauf1-andrea1.eyeS.rate = actual eyetracking sample rate
      - jbauf1-andrea1.fullrate = actual framerate of deinterlaced half-fields
      - jbauf1-andrea1.halfrate = actual framerate of interlaced full-fields
      - frames/ = raw video frames
	- frame000000.uyvy.bz2
  - replay/
    - mostly structured like test/ directory
    - frames/ - a symlink to the proper directory in ../../replay-clips
    - questions - post-viewing questions
    - answers - subject's replies to post-viewing questions

- analysis approach
  1) extract features from video frames
  2) learn correlation between features and eye positions
  3) use learned model to predict eye positions and compare with
     actual eye positions

- test-TopdownContext in toolkit
  sample scripts:
  1) /lab/tmpic/u/rob/home/rjpeters/science/projects/2005_topdown/20060605-slave.sh
  2) /lab/tmpic/u/rob/home/rjpeters/science/projects/2006_vgames/20070423-ghost-slave.sh
