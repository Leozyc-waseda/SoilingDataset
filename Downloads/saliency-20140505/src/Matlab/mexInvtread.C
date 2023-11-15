/*!@file Matlab/mexInvtread.C */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
// by the University of Southern California (USC) and the iLab at USC.  //
// See http://iLab.usc.edu for information about this project.          //
// //////////////////////////////////////////////////////////////////// //
// Major portions of the iLab Neuromorphic Vision Toolkit are protected //
// under the U.S. patent ``Computation of Intrinsic Perceptual Saliency //
// in Visual Environments, and Applications'' by Christof Koch and      //
// Laurent Itti, California Institute of Technology, 2001 (patent       //
// pending; application number 09/912,225 filed July 23, 2001; see      //
// http://pair.uspto.gov/cgi-bin/final/home.pl for current status).     //
// //////////////////////////////////////////////////////////////////// //
// This file is part of the iLab Neuromorphic Vision C++ Toolkit.       //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is free software; you can   //
// redistribute it and/or modify it under the terms of the GNU General  //
// Public License as published by the Free Software Foundation; either  //
// version 2 of the License, or (at your option) any later version.     //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is distributed in the hope  //
// that it will be useful, but WITHOUT ANY WARRANTY; without even the   //
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR      //
// PURPOSE.  See the GNU General Public License for more details.       //
//                                                                      //
// You should have received a copy of the GNU General Public License    //
// along with the iLab Neuromorphic Vision C++ Toolkit; if not, write   //
// to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,   //
// Boston, MA 02111-1307 USA.                                           //
// //////////////////////////////////////////////////////////////////// //
//
// Primary maintainer for this file: Rob Peters <rjpeters at usc dot edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Matlab/mexInvtread.C $
// $Id: mexInvtread.C 10360 2008-10-16 21:26:38Z itti $
//

#ifndef MATLAB_MEXINVTREAD_C_DEFINED
#define MATLAB_MEXINVTREAD_C_DEFINED

#include <mex.h>

#include "Matlab/mexConverts.H"
#include "Image/Image.H"
#include "Raster/GenericFrame.H"
#include "Raster/Raster.H"

#include <exception>

// ##########################################################################
//! Read any image into matlab
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  MYLOGVERB = LOG_INFO;
  LOG_FLAGS &= (~LFATAL_PRINTS_ABORT);
  LOG_FLAGS &= (~LFATAL_XTRA_NOISY);

  if (nrhs < 1)  mexErrMsgTxt("Missing required filename parameter");

  if (nlhs > 1)  mexErrMsgTxt("Too many output arguments (expected 0 or 1)");

  try
    {
      // get the file name:
      const int siz = mxGetN(prhs[0]) + 1;

      // (no need to mxFree() this pointer; matlab does that
      // automatically when the mex function returns)
      char* fname = (char*)mxCalloc(siz, sizeof(char));
      mxGetString(prhs[0], fname, siz);

      // read the image:
      const GenericFrame frame = Raster::ReadFrame(fname);

      // return the image if desired:
      if (nlhs > 0)
        {
          switch (frame.nativeType())
            {
            case GenericFrame::NONE:
              {
                Image<byte> empty;
                plhs[0] = Image2mexArrayUint8(&empty);
              }
              break;

            case GenericFrame::RGB_U8:
              {
                const Image<PixRGB<byte> > rgb(frame.asRgb());
                plhs[0] = RGBImage2mexArrayUint8(&rgb);
              }
              break;

            case GenericFrame::RGB_F32:
              {
                plhs[0] = RGBImage2mexArray(frame.asRgbF32());
              }
              break;

            case GenericFrame::GRAY_U8:
              {
                const Image<byte> gray(frame.asGray());
                plhs[0] = Image2mexArrayUint8(&gray);
              }
              break;

            case GenericFrame::GRAY_F32:
              plhs[0] = Image2mexArray(frame.asFloat());
              break;

            case GenericFrame::VIDEO:
              {
                // FIXME need a way to specify YUV dims if needed here:
                const Image<PixRGB<byte> > rgb(frame.asRgb());
                plhs[0] = RGBImage2mexArrayUint8(&rgb);
              }
              break;

            default:
              LFATAL("Cannot read frames of type %d", int(frame.nativeType()));
            }
        }
    }
  catch (std::exception& e)
    {
      mexErrMsgTxt(e.what());
    }
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // MATLAB_MEXINVTREAD_C_DEFINED
