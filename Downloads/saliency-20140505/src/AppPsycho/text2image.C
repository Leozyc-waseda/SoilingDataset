/*!@file AppPsycho/text2image.C convert sentences from a text file to
   images  */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the //
// University of Southern California (USC) and the iLab at USC.         //
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
// Primary maintainer for this file: David J. Berg <dberg@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/text2image.C $

#include "Component/ModelManager.H"
#include "Component/ComponentOpts.H"
#include "Image/Image.H"
#include "Image/DrawOps.H"
#include "Image/Pixels.H"
#include "Image/SimpleFont.H"
#include "Util/Types.H"
#include "Util/StringConversions.H"
#include "Util/StringUtil.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"

#include <fstream>

#define HDEG 54.9
// ######################################################################
extern "C" int main(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Create Images");

  nub::soft_ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "<textfile> visual-angle-of-single-character",
                               1, 2)==false)
    return(1);

  double fontsize = fromStr<double>(manager.getExtraArg(1));

  // let's get all our ModelComponent instances started:
  manager.start();

  // create an image frame for each sentence in our text file and store
  // it in a vector before we start the experiment, then we can just
  // present each frame like in psycho still
  //
  //First read the text file and all the sentences
  //load our file
  std::ifstream *itsFile;
  itsFile = new std::ifstream(manager.getExtraArg(0).c_str());

  //error if no file
  if (itsFile->is_open() == false)
    LFATAL("Cannot open '%s' for reading",manager.getExtraArg(0).c_str());

  //some storage variables
  std::string line;
  std::vector<std::vector<std::string> > lines;
  std::vector<uint> itsType;
  uint scount = 0;

  //loop through lines of file
  while (!itsFile->eof())
    {
      getline(*itsFile, line);

      std::vector<std::string> temp;
      //store the sentence and type (question or statement)
      if (line[0] == '#')//question
        {
          line.erase(0,1);
          temp.push_back(line);
          lines.push_back(temp);
          itsType.push_back(1);
          scount++;
        }
      else if (line[0] =='!')//sentence
        {
          line.erase(0,1);
          temp.push_back(line);
          lines.push_back(temp);
          itsType.push_back(0);
          scount++;
        }
      else
        {
          if (line.size() > 1)
            {
              scount--;
              lines[scount].push_back(line);
              scount++;
            }
        }
    }
  itsFile->close();

  //now we have stored all of our sentences, lets create our images
  int w = 1920;
  int h = 1080;
  uint fontwidth = uint(fontsize * w / HDEG);
  SimpleFont fnt = SimpleFont::fixedMaxWidth(fontwidth); //font
  std::vector<Image<PixRGB<byte> > > itsImage; //store sentences

  for (uint i = 0; i < lines.size(); i++)
    {
      int space = 0;
      int hanchor = int(h/2) - int(fnt.h()/2);
      Image<PixRGB<byte> > timage(w,h,ZEROS);
      PixRGB<byte> gr(128,128,128);
      timage += gr;

      for (uint j = 0; j < lines[i].size(); j++)
        {
          if (j < 1)
            space = int( double(w - fnt.w() * lines[i][j].size()) / 2.0 );
          if (j > 0)
            hanchor = hanchor + fnt.h();
          Point2D<int> tanchor(space, hanchor);
          writeText(timage,tanchor,lines[i][j].c_str(),
                    PixRGB<byte>(0,0,0),
                    gr,
                    fnt);
        }

      itsImage.push_back(timage);
    }


  //ok, now right out the image
  for (uint i = 0; i < itsImage.size(); i ++)
    {

      //check for void OFS
      if (ofs->becameVoid())
        {
          LINFO("quitting because output stream was closed or became void");
          return 0;
        }

      //update ofs
      const FrameState os = ofs->updateNext();

      //write out image
      ofs->writeRGB(itsImage[i], "output",
                    FrameInfo("Text imbeded image",SRC_POS));

      //last frame?
      if (os == FRAME_FINAL)
        break;

    }


  // stop all our ModelComponents
  manager.stop();

  // all done!
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
