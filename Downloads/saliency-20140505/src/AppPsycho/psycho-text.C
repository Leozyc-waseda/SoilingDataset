/*!@file AppPsycho/psycho-text.C Psychophysics display of text. Text
   elements are read in from a text file. Elements that require a
   response should be marked with a #at the beginning of the
   sentence.  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-text.C $

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/DrawOps.H"
#include "Image/SimpleFont.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "Util/Types.H"
#include "Util/StringConversions.H"
#include "Util/StringUtil.H"

#include <fstream>

#define HDEG 54.9
// ######################################################################
extern "C" int main(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Psycho Text");

  // Instantiate our various ModelComponents:
  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  nub::soft_ref<EyeTrackerConfigurator>
    etc(new EyeTrackerConfigurator(manager));
  manager.addSubComponent(etc);

  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);

  manager.setOptionValString(&OPT_EventLogFileName, "psychodata.psy");
  manager.setOptionValString(&OPT_EyeTrackerType, "ISCAN");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "<textfile> visual-angle-of-single-character",
                               1, 2)==false)
    return(1);
  double fontsize = fromStr<double>(manager.getExtraArg(1));

  // hook our various babies up and do post-command-line configs:
  nub::soft_ref<EyeTracker> et = etc->getET();
  d->setEyeTracker(et);
  d->setEventLog(el);
  et->setEventLog(el);

  // let's get all our ModelComponent instances started:
  manager.start();

  // let's do an eye tracker calibration:
  et->calibrate(d);

  d->clearScreen();
  d->displayText("<space> for random order, other key for ordered play.");
  int c = d->waitForKey();
  d->clearScreen();

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
  int w = d->getWidth();//width and height of SDL surface
  int h = d->getHeight();
  uint fontwidth = uint(fontsize * w / HDEG);
  SimpleFont fnt = SimpleFont::fixedMaxWidth(fontwidth); //font
  std::vector<Image<PixRGB<byte> > > itsSImage; //store sentences
  std::vector<Image<PixRGB<byte> > > itsQImage; //store images

  for (uint i = 0; i < lines.size(); i++)
    {
      int space = 0;
      int hanchor = int(h/2) - int(fnt.h()/2);
      Image<PixRGB<byte> > timage(w,h,ZEROS);
      timage += d->getGrey();

      for (uint j = 0; j < lines[i].size(); j++)
        {
          if (j < 1)
            space = int( double(w - fnt.w() * lines[i][j].size()) / 2.0 );
          if (j > 0)
            hanchor = hanchor + fnt.h();
          Point2D<int> tanchor(space, hanchor);
          writeText(timage,tanchor,lines[i][j].c_str(),
                    PixRGB<byte>(0,0,0),
                    d->getGrey(),
                    fnt);
        }
      if (itsType[i] == 0)
        itsSImage.push_back(timage);
      else
        itsQImage.push_back(timage);
    }

  uint count = scount/2;
  int indexS[count];
  int indexQ[count];
  for (uint i = 0; i < count; i ++)
    {
      indexS[i] = i;
      indexQ[i] = i;
    }

  if (c == ' ')
    {
      LINFO("Randomizing images...");
      randShuffle(indexS, count);
      randShuffle(indexQ, count);
    }

  bool nextSentence = true;
  uint ii =0;
  // main loop:
  for (uint i = 0; i < scount; i ++)
    {
      // load up the frame and show a fixation cross on a blank screen:
      d->clearScreen();
      Image< PixRGB<byte> > image;
      if (nextSentence)
        image = itsSImage[indexS[ii]];
      else
        image = itsQImage[indexQ[ii]];

      SDL_Surface *surf = d->makeBlittableSurface(image, true);

      if (nextSentence)
        LINFO("sentence '%d' ready.", ii);
      else
        LINFO("question '%d' ready.", ii);
      d->displayFixation();

      // ready to go whenever the user is ready:
      d->waitForKey();
      d->waitNextRequestedVsync(false, true);

      if (nextSentence)
        d->pushEvent(std::string("===== Showing Sentence: ") +
                     toStr<int>(indexS[ii])  + " =====");
      else
        d->pushEvent(std::string("===== Showing Question: ") +
                     toStr<int>(indexQ[ii])  + " =====");

      nextSentence = !nextSentence;
      if (i%2 != 0)
        ii++;

      // start the eye tracker:
      et->track(true);

      // blink the fixation:
      d->displayFixationBlink();

      // show the image:
      d->displaySurface(surf, -2);

      // wait for key:
      c = d->waitForKey();

      // free the image:
      SDL_FreeSurface(surf);

      // make sure display if off before we stop the tracker:
      d->clearScreen();

      // stop the eye tracker:
      usleep(50000);
      et->track(false);
    }

  d->clearScreen();
  d->displayText("Experiment complete. Thank you!");
  d->waitForKey();

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
