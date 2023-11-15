/*!@file AppPsycho/psycho-reading.C Psychophysics display of
   paragraphs, individtual sentences, and transitions between
   sentences with questions. */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-reading-eval.C $

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/ShapeOps.H"
#include "Image/DrawOps.H"
#include "Image/Transforms.H"
#include "Image/SimpleFont.H"
#include "Raster/Raster.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "Util/StringUtil.H"
#include "Util/sformat.H"

#include <iostream>
#include <fstream>

#define TEXTSCREEN 1920
#define TEXTYPOS 25
// ######################################################################
// Load our image files
// ######################################################################
std::vector<Image<PixRGB<byte> > > readParagraphs(const std::string& filename, 
                                                  std::vector<std::string>& names)
{
  names.clear();
  
  std::vector<std::string> parts;
  split(filename, ".", back_inserter(parts));
  const std::string base = parts[0];
  const std::string ext  = parts[1];
  
  std::vector<Image<PixRGB<byte> > > vec;
  int c = 0;
  bool add = false;
  std::string fn = filename;
  while (Raster::fileExists(fn))
    {
      //good file, so add it
      LINFO("adding file : %s:", fn.c_str());
      vec.push_back(Raster::ReadRGB(fn));
      names.push_back(fn);
      
      //next filename
      const int cp = (add) ? 1 : 0;
      fn = base + "_" + toStr<int>(c) + toStr<int>(c + cp) + "." + ext;
      if (add)
        ++c;
      add = !add;
    }
  return vec;
}

// ######################################################################
// count the number of summaries for a document
// ######################################################################
const uint countSummaries(const std::string& base)
{
  int ii = 1;
  std::string fn = base + "_" + toStr<int>(ii) + ".png";

  while (Raster::fileExists(fn))
    fn = base + "_" + toStr<int>(++ii) + ".png";
  
  return ii - 1;
}


// ######################################################################
extern "C" int main(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Psycho Reading");

  // Instantiate our various ModelComponents:
  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  nub::soft_ref<EyeTrackerConfigurator>
    etc(new EyeTrackerConfigurator(manager));
  manager.addSubComponent(etc);

  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);

  manager.setOptionValString(&OPT_EventLogFileName, "psychodata.psy");
  manager.setOptionValString(&OPT_EyeTrackerType, "EL");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "<questions file> <doc base1> <doc base2> ", 
                               2, -1)==false)
    return(1);

  // hook our various babies up and do post-command-line configs:
  nub::soft_ref<EyeTracker> et = etc->getET();
  d->setEyeTracker(et);
  d->setEventLog(el);
  et->setEventLog(el);
  
  // let's get all our ModelComponent instances started:
  manager.start();

  // setup array of document indices:
  uint nbimgs = manager.numExtraArgs()-1; int index[nbimgs];
  for (uint ii = 0; ii < nbimgs; ++ii) index[ii] = ii+1;
  randShuffle(index, nbimgs);

  SimpleFont fnt = SimpleFont::FIXED(20);

  /*read in the questions */
  std::vector<std::string > questions;
  std::ifstream file;
  file.open(manager.getExtraArg(0).c_str());
  if (file.is_open()) 
    {
      while (!file.eof()) 
        {
          std::string temp;
          getline(file, temp);
          LINFO("%s", temp.c_str());
          questions.push_back(temp);
        } 
      questions.pop_back();
      file.close();
    }
  else LFATAL("Cannot open questions file. ");
  
try
  {
    // main loop:
    for (uint i = 0; i < nbimgs; ++i)
      {
        // let's do an eye tracker calibration:
        LINFO("Have the subject read summary: %s", manager.getExtraArg(index[i]).c_str());
        d->displayText("Please read a summary <press enter when finished>, then we'll do a quick calibration.");
        d->waitForKey();

        LINFO("Performing calibration");
        et->calibrate(d);
        
        //clear the screen of any junk
        d->clearScreen();
        
        //randomize summaries for this document
        uint nbsums = countSummaries(manager.getExtraArg(index[i]));
        LINFO("%d Summaries found for Document '%s'", nbsums, 
              manager.getExtraArg(index[i]).c_str());
        int sindex[nbsums];
        for (uint ii = 0; ii < nbsums; ++ii) sindex[ii] = ii + 1;
        randShuffle(sindex, nbsums);
        
        for (uint j = 0; j < nbsums; ++j)
          {
            //write a loading message to the screen
            d->clearScreen();
            d->displayText("Loading - please wait.....");
            
            const std::string file = manager.getExtraArg(index[i]) + "_" + 
              toStr<int>(sindex[j]) + ".png";
            LINFO("Loading summary set '%s'...", file.c_str());
            std::vector<std::string> fnames;
            std::vector<Image<PixRGB<byte> > > imageset = 
              readParagraphs(file, fnames);
            
            //setup our anchors for displaying text
            //int w = imageset[0].getWidth();//width and height of SDL surface
            const int w = TEXTSCREEN;
            std::vector<Point2D<int> > anchors;
            std::vector<std::string >::const_iterator 
              iter(questions.begin()), end(questions.end());
            while (iter != end)
              {  
                int hposq = w/2 - 
                  int(double(fnt.w() * (uint)iter->size()) / 2.0);
                anchors.push_back(Point2D<int>(hposq, TEXTYPOS));
                ++iter;
              }
            
            d->clearScreen();
            
            //loop over the images
            for (uint k = 0; k < imageset.size(); ++k)
              {
                //display the first image
                Image<PixRGB<byte> > cimg = imageset[k];
                
                //if our first image display with fixation blink etc              
                if (k == 0)
                  {
                    //create surface
                    SDL_Surface *surf = d->makeBlittableSurface(cimg, true);
                    
                    LINFO("ready '%s'...", fnames[k].c_str());
                    
                    //do a quick drift correct
                    et->recalibrate(d,0);
                    
                    //fixation 
                    //d->displaFixation();
                    
                    // ready to go whenever the user is ready:
                    //d->waitForKey();
                    d->waitNextRequestedVsync(false, true);
                    d->pushEvent(sformat("===== Showing text: %s =====", 
                                         fnames[k].c_str()));
                    
                    // start the eye tracker:
                    et->track(true);
                    
                    // blink the fixation:
                    d->displayFixationBlink();
                    
                    // show the image:
                    d->displaySurface(surf, -2);
                    
                    // wait for key:
                    d->waitForKey();
                    
                    // free the image:
                    SDL_FreeSurface(surf);
                    
                    // stop the eye tracker:
                    usleep(50000);
                    et->track(false);
                    LINFO("recorded eye tracker session");
                  }
                
                //ask the user a questions and record a new eye tracker session:
                std::vector<Image<PixRGB<byte> > > cimgq(questions.size(), cimg);
                for (uint m = 0; m < questions.size(); ++m)
                  {
                    //create images with questions 
                    const float factor = (float)d->getHeight() / (float)d->getWidth();
                    const int tempv = TEXTSCREEN * factor;
                    Image<PixRGB<byte> > temp(TEXTSCREEN, tempv, ZEROS);
                    
                    writeText(temp, anchors[m], questions[m].c_str(), 
                              PixRGB<byte>(255,255,255), PixRGB<byte>(0,0,0),fnt);
                    
                    //combine question text with our current image
                    temp = rescaleBilinear(temp, cimgq[m].getDims());
                    cimgq[m] = composite(temp, cimgq[m], PixRGB<byte>(0,0,0));
                    
                    //create surface
                    SDL_Surface *surf = d->makeBlittableSurface(cimgq[m], true);
                    
                    LINFO("ready Q%d '%s'...", m, fnames[k].c_str());
                    
                    d->waitNextRequestedVsync(false, true);
                    d->pushEvent(sformat("===== Showing Q%d \"%s\": %s =====", 
                                         m, questions[k].c_str(), fnames[k].c_str()));
                    
                    // show the image:
                    d->displaySurface(surf, -2);
                    d->waitForKey();
                    SDL_FreeSurface(surf);
                  }
              }
          }
      }
    
    d->clearScreen();
    d->displayText("Experiment complete. Thank you! Press any key.");
    d->waitForKey();
    
  }
 catch (...)
   {
     REPORT_CURRENT_EXCEPTION;
   };
 
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
