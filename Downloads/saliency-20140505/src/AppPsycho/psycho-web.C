/*!@file AppPsycho/psycho-web.C displays a text message, and then
   opens a web page where the subjects eyes are tracked., afterwords,
   displays a question? The web browser must be opened and
   appropriately sized first. */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-web.C $

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/DrawOps.H"
#include "Image/SimpleFont.H"
#include "Raster/Raster.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "Util/Types.H"
#include "Util/StringConversions.H"
#include "Util/StringUtil.H"
#include "rutz/pipe.h"

#include <fstream>
#include <X11/X.h>
#include <X11/Xlib.h>
#include <X11/Intrinsic.h>
#include <X11/StringDefs.h>
#include <X11/Xutil.h>
#include <X11/Shell.h>

#define HDEG 54.9
#define KEY_BUFF_SIZE 256


// #####################################################################
// some code to collect clicks from all xwindows
// ######################################################################
void TranslateKeyCode(XEvent *ev, char* key_buff)
{
  int count;
  KeySym ks;

  if (ev)
   {
     count = XLookupString((XKeyEvent *)ev, key_buff, KEY_BUFF_SIZE, &ks,NULL);
     key_buff[count] = '\0';

     if (count == 0)
       {
         char *tmp = key_buff = XKeysymToString(ks);
         if (tmp)
           strcpy(key_buff, tmp);
         else
           key_buff[0] = '\0';
       }
   }
  else
    key_buff[0] = '\0';
}

// ######################################################################
void snoop_all_windows(Window root, unsigned long type, Display* d)
{
  static int level = 0;
  Window parent, *children;
  unsigned int nchildren;
  int stat;

  level++;

  stat = XQueryTree(d, root, &root, &parent, &children, &nchildren);
  if (stat == FALSE)
   {
     LINFO("Can't query window tree...\n");
     return;
   }

  if (nchildren == 0)
    return;

  /* For a more drastic inidication of the problem being exploited
   * here, you can change these calls to XSelectInput() to something
   * like XClearWindow(d, children[i]) or if you want to be real
   * nasty, do XKillWindow(d, children[i]).  Of course if you do that,
   * then you'll want to remove the loop in main().
   *
   * The whole point of this exercise being that I shouldn't be
   * allowed to manipulate resources which do not belong to me.
   */
  XSelectInput(d, root, type);

  for(uint i=0; i < nchildren; i++)
   {
     XSelectInput(d, children[i], type);
     snoop_all_windows(children[i], type, d);
   }

  XFree((char *)children);
}

// ######################################################################
void waitXKeyPress(char *hostname)
{
  XEvent xev;
  char key_buff[KEY_BUFF_SIZE];
  key_buff[0] = '\0';

  Display *d = NULL;
  d = XOpenDisplay(hostname);
  if (d == NULL)
   {
     LFATAL("Blah, can't open display: %s\n", hostname);
   }

  snoop_all_windows(DefaultRootWindow(d), KeyPressMask, d);

  while(1)
   {
     XNextEvent(d, &xev);
     TranslateKeyCode(&xev, key_buff);
     LINFO("polling");
     if (strlen(key_buff)>0)
       break;
   }
}

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
                               "<textfile> <webpage> "
                               "<visual-angle-of-single-character>",
                               3, 3)==false)
    return(1);

  //get degrees
  double fontsize = fromStr<double>(manager.getExtraArg(2));
  //get webpage
  std::string itsWeb = manager.getExtraArg(1);

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
  int index[count];
  for (uint i = 0; i < count; i ++)
    {
      index[i] = i;
    }

  if (c == ' ')
    {
      LINFO("Randomizing images...");
      randShuffle(index, count);
    }

  char* display = getenv((char*)"DISPLAY");
  // main loop:
  for (uint i = 0; i < count; i ++)
    {
      // load up the frame and show a fixation cross on a blank screen:
       d->clearScreen();

       //seutp sdl surfaces
      Image< PixRGB<byte> > imageS = itsSImage[index[i]];
      SDL_Surface *surf = d->makeBlittableSurface(imageS, true);

      Image< PixRGB<byte> > imageQ = itsQImage[index[i]];
      SDL_Surface *surfq = d->makeBlittableSurface(imageQ, true);

      // ready to go whenever the user is ready:
      d->displayFixation();
      d->waitForKey();
      d->waitNextRequestedVsync(false, true);

      d->pushEvent(std::string("===== Showing Sentence: ") +
                   toStr<int>(index[i])  + " =====");

      // show the image:
      d->displaySurface(surf, -2);
      d->waitForKey();
      // free the image:
      SDL_FreeSurface(surf);
      d->clearScreen();
      d->waitNextRequestedVsync(false, true);

      rutz::exec_pipe open("r","/usr/bin/mozilla-firefox",
                           itsWeb.c_str(), NULL);

      // start the eye tracker:
      et->track(true);

      // blink the fixation:
      d->displayFixationBlink();

      //open webpage - close sdl
      d->closeDisplay();
      d->pushEvent(std::string("===== Showing web page: ") +
                toStr<int>(index[i])  + " =====");

      //wait for a key in any window to be pressed
      waitXKeyPress(display);

      // stop the eye tracker:
      usleep(50000);
      et->track(false);
      d->pushEvent(std::string("=====  Destroying web page: ") +
                toStr<int>(index[i])  + " =====");

      //open sdl again
      d->openDisplay();

      //display the question
      d->pushEvent(std::string("===== Showing Question: ") +
                   toStr<int>(index[i])  + " =====");

      // show the image:
      d->displaySurface(surfq, -2);
      d->waitForKey();
      // free the image:
      SDL_FreeSurface(surfq);
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
