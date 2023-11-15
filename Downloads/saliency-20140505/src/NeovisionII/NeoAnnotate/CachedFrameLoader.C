/*!@file NeovisionII/NeoAnnotate/CachedFrameLoader.C Cached interface to InputFrameSeries */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/NeoAnnotate/CachedFrameLoader.C $
// $Id: CachedFrameLoader.C 13844 2010-08-30 21:00:19Z rand $
//

#include "NeovisionII/NeoAnnotate/CachedFrameLoader.H"
#include "Component/ModelComponent.H"
#include "Image/Image.H"
#include "Image/PixelsTypes.H"
#include "QtUtil/ImageConvert4.H"

CachedFrameLoader::CachedFrameLoader() :
  itsVideoDecoder(NULL),
  itsDims(0,0)
{
  // Set our cache size, and clear the cache:
  int cacheSize_mb = 1000;
  QPixmapCache::setCacheLimit(cacheSize_mb * 1024);
  QPixmapCache::clear();
}

QImage CachedFrameLoader::getFrame(int frameNum)
{
  if(itsVideoDecoder == NULL) return QImage();

  //Tell the input frameseries to seek to the current frame
  itsVideoDecoder->setFrameNumber(frameNum);

  //Read the current frame, convert it, and cache it
  return convertToQImage4(itsVideoDecoder->readFrame().asRgb());

//  //Create a hash key for the requested frame (just the framenumber in string form)
//  QString key = QString("%1").arg(frameNum);
//  QImage ret;
//
//  //Try to look up the frame in our frame cache
//  if(!QPixmapCache::find(key, ret))
//  {
//    //Tell the input frameseries to seek to the current frame
//    itsIfs->setFrameNumber(frameNum);
//
//    //Read the current frame, convert it, and cache it
//    ret = convertToQPixmap4(itsIfs->readRGB());
//
//    //Insert the frame into the cache
//    QPixmapCache::insert(key, ret);
//  }
//
//  return ret;
}

Dims CachedFrameLoader::getDims() const
{
  return itsDims;
}

FrameRange CachedFrameLoader::getFrameRange() const
{
  if(itsVideoDecoder == NULL) return FrameRange(0,0,0);
  return FrameRange(0, 1, itsVideoDecoder->getNumFrames());
}

bool CachedFrameLoader::loadVideo(QString filename)
{
  itsDims = Dims(0,0);
  if(itsVideoDecoder) delete itsVideoDecoder;

  itsVideoDecoder = new MgzJDecoder(filename.toStdString().c_str());

  itsVideoDecoder->setFrameNumber(0);
  itsDims = itsVideoDecoder->readFrame().getDims();

  return true;
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

