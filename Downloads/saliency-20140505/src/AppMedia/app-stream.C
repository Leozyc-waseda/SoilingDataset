/*!@file AppMedia/app-stream.C simple program to exercise FrameIstream
  and FrameOstream */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/app-stream.C $
// $Id: app-stream.C 9841 2008-06-20 22:05:12Z lior $
//

#ifndef APPMEDIA_APP_STREAM_C_DEFINED
#define APPMEDIA_APP_STREAM_C_DEFINED

#include "Component/ModelManager.H"
#include "Media/Streamer.H"

/* This is a trivial program bin/stream to exercise FrameIstream and
   FrameOstream. All it does is read an InputFrameSeries, but you can
   specify the input to come from anywhere (using --in) and you can
   specify the output to go anywhere (using one or more --out
   options). E.g., to view an mpeg movie (note that we make no effort
   to actually display the movie at the proper framerate):

     ./bin/stream --input-frames=0-10000@1 \
       --in=tests/inputs/mpegclip1.mpg --out=display

   to display the contents of your framegrabber (again, we make no
   effort to sync up the framerates)

     ./bin/stream --input-frames=0-10000@1 --in=v4l --out=display

   to convert a movie into a series of raster files (the output files
   will be named starting from stream-output000000.ppm):

     ./bin/stream --input-frames=0-10000@1 \
       --in=tests/inputs/mpegclip1.mpg --out=raster

   to record 100 frames from your your framegrabber to a movie file
   (the movie will be named stream-output.mpg):

     ./bin/stream --input-frames=0-99@1 --in=v4l --out=mpeg

   to view some white noise in case you are bored:

     ./bin/stream --input-frames=0-999999@1 \
       --in=random:1024x1024 --out=display
*/

class CopyStreamer : public Streamer
{
public:
  CopyStreamer()
    :
    Streamer("Streamer"),
    itsOutPrefix("stream-output")
  {}

private:
  virtual void handleExtraArgs(const ModelManager& mgr)
  {
    if (mgr.numExtraArgs() > 0)
      itsOutPrefix = mgr.getExtraArg(0);
  }

  virtual void onFrame(const GenericFrame& input,
                       FrameOstream& ofs,
                       const int frameNum)
  {
    ofs.writeFrame(input, itsOutPrefix,
                   FrameInfo("copy of input frame", SRC_POS));
  }

  std::string itsOutPrefix;
};

int main(const int argc, const char **argv)
{
  CopyStreamer s;
  return s.run(argc, argv, "?prefix?", 0, 1);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif // APPMEDIA_APP_STREAM_C_DEFINED
