/*!@file Matlab/mexLog.H a few helpful functions for logging in mex files
 */
// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2002   //
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
// Primary maintainer for this file: Dirk Walther <walther@caltech.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Matlab/mexLog.H $
// $Id: mexLog.H 6003 2005-11-29 17:22:45Z rjpeters $
//

#ifndef MEXLOG_H_DEFINED
#define MEXLOG_H_DEFINED

#define FPEXCEPTIONSDISABLED
#include <streambuf>
#include <mex.h>

//! writes an info message to mexPrintf
/*! to get printf-style formatting, use sformat() from
    "Util/sformat.H" */
void mexInfo(const std::string& msg);

//! writes an error message
/*! to get printf-style formatting, use sformat() from
    "Util/sformat.H" */
void mexError(const std::string& msg);

//! writes an error message and leaved the mex file
/*! to get printf-style formatting, use sformat() from
    "Util/sformat.H" */
void mexFatal(const std::string& msg);

//! writes a debug message
/*! to get printf-style formatting, use sformat() from
    "Util/sformat.H" */
void mexDebug(const std::string& msg);

// ######################################################################
//! This streambuf is used to re-direct std::cout and std::cerr
class MexBuf : public std::streambuf
{
public:
  MexBuf()  : std::streambuf()
  { setbuf((char*)0,0); }

  virtual int overflow(int c)
  {
    if (c != EOF)
      mexPrintf("%c", c);
    return c;
  }

  virtual std::streamsize xsputn(const char* s, const std::streamsize n)
  {
    std::streamsize c = n;
    while (*s && c--)
      mexPrintf("%c", *s++);
    return n;
  }

};

// ######################################################################
//! This class triggers the redirection of the standard streams at construction
class MexBufInit
{
public:
  MexBufInit(MexBuf& buf);

  ~MexBufInit();

private:
  MexBufInit(const MexBufInit&);
  MexBufInit& operator=(const MexBufInit&);
};

// ######################################################################
namespace
{
  static MexBuf mexbuf__;
  static MexBufInit mexbufInit__(mexbuf__);
}

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
