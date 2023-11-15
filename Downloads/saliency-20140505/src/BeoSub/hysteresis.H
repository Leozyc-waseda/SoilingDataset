/*! @file BeoSub/hysteresis.H [put description here] */

// Function declarations for hysteresis.H

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/hysteresis.H $
// $Id: hysteresis.H 6182 2006-01-31 18:41:41Z rjpeters $

int apply_hysteresis(short int *mag, unsigned char *nms, int rows, int cols,
                     float tlow, float thigh, unsigned char *edge);

void non_max_supp(short *mag, short *gradx, short *grady, int nrows, int ncols,
                  unsigned char *result);
