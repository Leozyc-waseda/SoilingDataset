/*!@file CINNIC/gaborElement.H A Gabor element */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CINNIC/gaborElement.H $
// $Id: gaborElement.H 4663 2005-06-23 17:47:28Z rjpeters $

// ############################################################
// ############################################################
// ##### ---CINNIC---
// ##### Contour Integration:
// ##### T. Nathan Mundhenk nathan@mundhenk.com
// ############################################################
// ############################################################


class gaborElement
{
public:
  gaborElement();
  ~gaborElement();
  //! optional item ID number for gabor
  long item;
  //! x,y position of gabors
  int x,y;
  //! standard deviation
  float stddev;
  //! gabor period
  float period;
  //! gabor phase
  float phase;
  //! gabor theta (angle)
  float theta;
  //! use 1.0F or 2.0F depending on study
  /*!(e.g. Polat and Sagi, 1994 = 1, Braun, 1999 = 2)*/
  float sigMod;
  //! the amplitude of the gabor (a constant A*gabor)
  float amplitude;
  //! Tells if this is a master element (to be detected) or not
  int master;
};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
