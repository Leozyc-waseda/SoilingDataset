/*!@file NeovisionII/NeoAnnotate/CachedFrameLoader.H A Cached interface to InputFrameSeries */

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
// Primary maintainer for this file:
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/NeoAnnotate/CachedFrameLoader.H $
// $Id: CachedFrameLoader.H 13844 2010-08-30 21:00:19Z rand $
//

#ifndef NEOVISIONII_NEOANNOTATE_CACHEDFRAMELOADER_H_DEFINED
#define NEOVISIONII_NEOANNOTATE_CACHEDFRAMELOADER_H_DEFINED

#include "Component/OptionManager.H"
#include "Component/ModelComponent.H"
#include "Media/MgzJDecoder.H"
#include "Media/FrameSeries.H"

#include <QtGui/QtGui>

//! Cached interface to InputFrameSeries which directly delivers QImage images
class CachedFrameLoader: public ModelComponent
{
public:
  //! Constructor
  CachedFrameLoader();

  //! Grab a new frame from the frame series, and convert it to a QImage
  QImage getFrame(int frameNum);

  //! Get the dimensions of the video frames
  Dims getDims() const;

  //! Get the framerange of the movie
  FrameRange getFrameRange() const;
    
  std::string getFrameFilename(int fnum);

  bool loadVideo(QString filename);


private:
  MgzJDecoder* itsVideoDecoder;
  std::string itsFilenameRoot;
  Dims itsDims;
};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // NEOVISIONII_NEOANNOTATE_CACHEDFRAMELOADER_H_DEFINED
