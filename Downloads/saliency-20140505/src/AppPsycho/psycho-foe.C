/*!@file AppPsycho/psycho-search.C Psychophysics display for a search for a
  target that is presented to the observer prior to the search */
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
// The iLab Neuromorphic Vision C++ Toolkit is free software; you can   //
// redistribute it and/or modify it under the terms of the GNU General  //
// Public License as published by the Free Software Foundation; either  //
// version 2 of the License, or (at your option) any later version.     //
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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-foe.C $
// $Id: psycho-foe.C 12962 2010-03-06 02:13:53Z irock $

#include "Component/ModelManager.H"
#include "Image/ColorOps.H" // for makeRGB()
#include "Image/CutPaste.H" // for inplacePaste()
#include "Image/Image.H"
#include "Image/MathOps.H"  // for inplaceSpeckleNoise()
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "Raster/Raster.H"
#include "Util/MathFunctions.H"
#include "Util/StringUtil.H"
#include "Util/Types.H"
#include "Video/VideoFrame.H"
#include "rutz/time.h"
#include "Raster/Raster.H"
#include <deque>
#include "Media/MPEGStream.H"
#include "Media/MediaOpts.H"

#include <ctype.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#define CACHELEN 150
#define NMASK 10

using namespace std;
//! number of frames in the mask

// ######################################################################
static bool cacheFrame(nub::soft_ref<InputMPEGStream>& mp,
                       std::deque<VideoFrame>& cache)
{
  const VideoFrame frame = mp->readVideoFrame();
  if (!frame.initialized()) return false; // end of stream
  cache.push_front(frame);
  return true;
}

static int submain(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages
  ModelManager manager("Psycho Movie");
  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);
  nub::soft_ref<EyeTrackerConfigurator> etc(new EyeTrackerConfigurator(manager));
  manager.addSubComponent(etc);
  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);
  nub::soft_ref<InputMPEGStream> mp (new InputMPEGStream(manager, "Input MPEG Stream", "InputMPEGStream"));
  manager.addSubComponent(mp);
  manager.setOptionValString(&OPT_InputMPEGStreamPreload, "true");

  // Parse command-line:
  // Subject - MaleOrFemale -  pilot - Experiment Number - Run Number - Movie Type
  if (manager.parseCommandLine(argc, argv," We have 6 args:  Subject - MaleOrFemale -  pilot - Experiment Number - Run Number - Movie Type ", 6, 6)==false){
        cout<<"please give the following arguments:\n 1)Subject-Initials 2)M/F 3)Is-Pilot(1 or 0) 4)ExpSetNumber 5)Run-Number 6)movie type(1,2,3)"<< endl;
             return(1);
        }
        string Sub=manager.getExtraArg(0);
        string Gender=manager.getExtraArg(1);
        string pilot=manager.getExtraArg(2);
        int isPilot=(int)pilot[0]-48;//
                cout<<isPilot<<"is a pilot experiment"<<endl;
        string experiment_set_number=manager.getExtraArg(3);
        int experiment_set_no=(int)experiment_set_number[0]-48;//atoi(PT[0].c_str());
                cout<<"experiment set number is:"<<experiment_set_no<<endl;
        string Run=manager.getExtraArg(4);
        string DataDir="/home2/tmp/u/elno/research/exp1/data/";
        string psyFileName=DataDir + Sub + "_" + Gender + "__" + pilot +"__Set_Number--"+ experiment_set_number+"__RUN--"+Run;
        string movie_type=manager.getExtraArg(5);
        //int bg_type=(int)movie_type[0]-48;//atoi(PT[0].c_str());

        manager.setOptionValString(&OPT_EventLogFileName, psyFileName);
        manager.setOptionValString(&OPT_EyeTrackerType, "ISCAN");

        // Push Events for this one :-)
        d->pushEvent(std::string("Run=") + Run + "\nSubject=" + Sub +"\nGender="+ Gender +"\n" );
        d->pushEvent(std::string("Set=") + experiment_set_number + "\n");
        d->pushEvent(std::string("MovieType=") + movie_type + "\n");
        if (isPilot) {d->pushEvent("Piloting");}
        else {d->pushEvent("Control");}

        nub::soft_ref<EyeTracker> et = etc->getET();
        d->setEyeTracker(et);
        d->setEventLog(el);
        et->setEventLog(el);

        if (etc->getModelParamString("EyeTrackerType").compare("EL") == 0)
        d->setModelParamVal("SDLslaveMode", true);
        manager.start();


// Getting the foreground images from the set we needed

   string imagelisthomeDir="/home2/tmp/u/elno/research/exp1/stim/fgImages";
   string imagelistsetFile=imagelisthomeDir+"/set"+experiment_set_number.c_str()+"/imagelist";

          cout<<"Trying to open imagelist : "<<imagelistsetFile<< endl;

  FILE *f = fopen(imagelistsetFile.c_str(), "r");
  if (f == NULL) LFATAL("Cannot read stimulus file");
  char line[1024];
  std::vector<std::string> ilist, tlist, rlist, slist, vlist;

// Get the FOE movie directory
        string foeDir="/home2/tmp/u/elno/research/exp1/stim/bgMovies/foe/";
        for (int foe=1; foe<6; foe++){
                std::string s;
                std::stringstream out;
                s="";
                out << foe;
                s = out.str();
                string foeFileName=foeDir + s + "/";
                vlist.push_back(foeFileName); // FOE movies back ground
        }

      LINFO("\n video : %s,",vlist[0].c_str());
      LINFO("\n video : %s,",vlist[1].c_str());
      LINFO("\n video : %s,",vlist[2].c_str());
      LINFO("\n video : %s,",vlist[3].c_str());
      LINFO("\n video : %s,",vlist[4].c_str());

  int correct = 0, accuracy = 100, total = 0;
  while(fgets(line, 1024, f)){
        std::vector<std::string> tokens;
              LINFO("line reads %s",line);
              split(line," ", std::back_inserter(tokens));
      // now line is at the imagelet, line2 at the image
      tlist.push_back(std::string(tokens[0])); // ONLY THE TARGET
      ilist.push_back(std::string(tokens[1])); // IMAGE WITH TARGET
      rlist.push_back(std::string(tokens[2])); // Image to report Position of the target
      slist.push_back(std::string(tokens[3])); // spec file for the search array

      LINFO("\nNew pair \nline1 reads: %s, \nline2 reads:%s, \nline 3 reads %s,\nline 4 reads %s",tokens[0].c_str(), tokens[1].c_str(),tokens[2].c_str(), tokens[3].c_str());
    }
  fclose(f);
  // randomize stimulus presentation order:
  int nimg = ilist.size(); int imindex[nimg];
  for (int i = 0; i < nimg; i ++) imindex[i] = i;
       randShuffle(imindex, nimg);

  int n_vids = ilist.size();
  int vid_index[n_vids];
  for (int i = 0; i < n_vids; i ++) vid_index[i] = i;
       randShuffle(vid_index, n_vids);

  for (int i = 0; i < n_vids; i ++)
        vid_index[i] = vid_index[i]%5;
  // let's display an ISCAN calibration grid:
  d->clearScreen();
  d->displayISCANcalib();
  d->waitForKey();
  // let's do an eye tracker calibration:
  d->displayText("<SPACE> to calibrate; other key to skip");

  int c = d->waitForKey();
  if (c == ' ') d->displayEyeTrackerCalibration(3, 3);

  //  if (c == ' ') d->displayEyeTrackerCalibration(3, 3);
/*
  if (c == ' ')
    {
      et->track(true);
      et->calibrateOnline(d);
      et->track(false);
    }
*/

  // we are ready to start:
  d->clearScreen();
  d->displayText("<SPACE> to start experiment");
  d->waitForKey();
  //******************* main loop:***********************//
  for (int trial = 0; trial < nimg; trial++) {
        // Trial information push down
                std::string stt;
                std::stringstream outt;
                outt << trial;
                stt = outt.str();
                d->pushEvent(std::string("===== set number / trial number target image:  ") + experiment_set_number + std::string(" === ")+ stt + std::string(" ====="));

        int imnum = imindex[trial];
             int v_num = vid_index[trial];

                LINFO("Loading array image: '%s' ... /target: '%s' .../report: '%s'... /video: '%s' ... /spec: '%s'   ", ilist[imnum].c_str(), tlist[imnum].c_str(), rlist[imnum].c_str(), vlist[v_num].c_str(), slist[imnum].c_str());

        // load up the images and show a fixation cross on a blank screen:
        // get the imagelet and place it at a random position:
                if(!Raster::fileExists(tlist[imnum]))
                {
                        manager.stop();
                        LFATAL("i couldnt find target image file %s", tlist[imnum].c_str());
                }
        //Get target image
        Image< PixRGB<byte> > img = Raster::ReadRGB(tlist[imnum]);
        Image< PixRGB<byte> > rndimg(d->getDims(), NO_INIT);
        rndimg.clear(d->getGrey());
        int rndx = 0,rndy = 0;
        SDL_Surface *surf1 = d->makeBlittableSurface(img, true);
        char buf[256];
                if(!Raster::fileExists(ilist[imnum]))
                {
                        manager.stop();
                        LFATAL("i couldnt find array image file %s", ilist[imnum].c_str());
                }
        img = Raster::ReadRGB(ilist[imnum]);
        SDL_Surface *surf2 = d->makeBlittableSurface(img, true);
        //randShuffle(mindex, NMASK);
        // load up the reporting number image:
                if(!Raster::fileExists(rlist[imnum]))
                {
                        manager.stop();
                        LFATAL(" i couldnt find report image file %s", rlist[imnum].c_str());
                }

        img = Raster::ReadRGB(rlist[imnum]);
        SDL_Surface *surf3 = d->makeBlittableSurface(img, true);

        // give a chance to other processes if single-CPU:
    d->clearScreen();
    usleep(200000);
    // ready to go whenever the user is ready:
    d->displayFixationBlink();
    //d->waitForKey();
    d->waitNextRequestedVsync(false, true);

    //************************ Display target************************//
    sprintf(buf, "===== Showing imagelet: %s at (%d, %d) =====", tlist[imnum].c_str(), rndx, rndy);
    d->pushEvent(std::string("===== Showing target image: ") + tlist[imnum] + std::string(" ====="));
    d->pushEvent(buf);
 if (trial<5){
            d->displayText("Find this target among others:");
            usleep(2000000);
}
    d->displaySurface(surf1, 0, true);
    usleep(3000000);
    d->clearScreen();
    //************************ Display array************************//
      LINFO("Buffering foe video '%s'...",(vlist[v_num]).c_str());
     // LINFO("Loading %s", manager.getExtraArg(ilist[imnum]).c_str());
      Image<PixRGB<byte> > cimage = Raster::ReadRGB(ilist[imnum]);
      //transparent pixel should be in upper left hand corner
      PixRGB<byte> currTransPixel = cimage[0];
      // cache initial movie frames:
      bool streaming = true;
      LINFO("Buffering '%s'...",(vlist[v_num]).c_str());
      int video_index= int(9.0 * randomDouble())+1;
        std::string sv="";
        std::stringstream outv;
        outv << video_index;
        sv = outv.str();
        sv=vlist[v_num].c_str()+movie_type+"-"+sv+".mpg";
      mp->setFileName(sv.c_str());
      std::deque<VideoFrame> cache;
      for (uint j = 0; j < CACHELEN; j ++)
        {
          streaming = cacheFrame(mp, cache);
          if (streaming == false) break;  // all movie frames got cached!
        }
      // give a chance to other processes (useful on single-CPU machines):
      sleep(1); system("/bin/sync");
      // display fixation to indicate that we are ready:
      d->displayFixation();
      // ready to go whenever the user is ready:
      d->waitForKey(true); int frame = 0;
      d->waitNextRequestedVsync(false, true);
      d->pushEvent(std::string("===== Playing movie: ") + sv.c_str() + " =====");
      // start the eye tracker:
      et->track(true);
      // blink the fixation:
      d->displayFixationBlink();
      // show the image:
      d->pushEvent(std::string("===== Showing search image: ") + ilist[imnum] + std::string(" ====="));
      // create an overlay:
      d->createVideoOverlay(VIDFMT_YUV420P,mp->getWidth(),mp->getHeight()); // mpeg stream returns YUV420P
     // play the movie:

      rutz::time start = rutz::time::wall_clock_now();

    Timer timer(100);
    timer.reset();
    double startTime=timer.getSecs();
    double EndTime=timer.getSecs();

    // wait for key; it will record reaction time in the logs:

    bool timeUp=false;

      while(cache.size() && d->checkForKey()<0 &&  !timeUp ) {
        if(timer.getSecs() > startTime +10)
        {
          timeUp=true;
          d->pushEvent(std::string("===== Time Up ====="));
        }
          if (streaming) streaming = cacheFrame(mp, cache);
          VideoFrame vidframe = cache.back();
          d->displayVideoOverlay_image(vidframe, frame,
                                       SDLdisplay::NEXT_VSYNC,
                                       cimage, currTransPixel,4);
          cache.pop_back();
          ++frame;

        EndTime=timer.getSecs();
      }

      rutz::time stop = rutz::time::wall_clock_now();
      const double secs = (stop-start).sec();
      const double responsetime = EndTime - startTime;

        std::string stdob;
        std::stringstream outdob;
        outdob << secs;
        stdob = outdob.str();

        std::string stdob2;
        std::stringstream outdob2;
        outdob2 << responsetime;
        stdob2 = outdob2.str();


      d->pushEvent(std::string("==== Total Time:  ") + stdob + std::string(" =====") );
      d->pushEvent(std::string("==== Response Time:  ") + stdob2 + std::string(" =====") );

      LINFO("%d frames in %.02f sec (~%.02f fps)", frame, secs, frame/secs);
      d->destroyYUVoverlay();

      et->track(false); //we just want to record eye movements while subjects view the array   // stop the eye tracker:
      d->clearScreen();  // sometimes 2 clearScreen() are necessary
      d->clearScreen();
      usleep(2000000);
      d->displayText("Use this image to report target's position: ");
      usleep(2000000);
    //d->waitForKey();
    //*********************** Display reporting image**************//
    // show the reporting image:
    d->pushEvent(std::string("===== Showing reporting image: ") + rlist[imnum] + std::string(" ====="));
    d->displaySurface(surf3, 0, true);
    usleep(3000000);
    usleep(5000);
    d->displayText("Input the target number: \n\n\n [press enter]");
    string inputString = d->getString('\n');
    //check user response
    char tmp[40];
    //lets open the spec file and extract the target number


        string buffer=slist[imnum].c_str();
        int s = buffer.size();
        string::iterator it;
        for(int x = 0; x < s ; x++)
        {
                if(buffer.at(x) == '\n')
                {
                        it = buffer.begin() + x;
                        buffer.erase(it);
                        s--;                                                        // Correction of string length
                }
        }


    ifstream specFile(buffer.c_str(), ifstream::in);
    bool found =false;
    string testLine;
    std::vector<std::string> specTokens;

   if(specFile.is_open())
      {
        while(!specFile.eof() && !found)
          {
            getline(specFile, testLine);
            string::size_type loc = testLine.find("target", 0);

            if(loc != string::npos)
               found = true;
          }
        if(!found)
          {
            manager.stop();
            LFATAL("couldnt find the target number from spec file");
          }

        split(testLine," ", std::back_inserter(specTokens));

        std::string responseString;
        responseString = inputString;
        int intResponse = atoi(responseString.c_str()), intActual = atoi(specTokens[1].c_str());

        total++;

        if (intResponse == intActual)
            {
              correct ++;
              accuracy = correct * 100 / total;
              sprintf(tmp, "Correct! Accuracy is %d%%", accuracy);
              d->displayText(tmp);
              d->pushEvent(std::string("===== Correct ====="));
              usleep(500000);
            }
          else
            {
              accuracy = correct * 100 / total;
              sprintf(tmp, "Wrong! Accuracy is %d%%", accuracy);
              d->displayText(tmp);
              d->pushEvent(std::string("===== Wrong ====="));
              usleep(500000);
            }

        specFile.close();
      }
    else
      {
        d->displayText("no target file found!");
        LFATAL("couldnt open the file -%s-",slist[imnum].c_str());
      }

    // free the imagelet and image:
    SDL_FreeSurface(surf1); SDL_FreeSurface(surf2); SDL_FreeSurface(surf3);

    // let's do a quiinckie eye tracker calibration once in a while:
    /*if (im > 0 && im % 20 == 0) {
      d->displayText("Ready for quick recalibration");
      d->waitForKey();
      d->displayEyeTrackerCalibration(3, 3);
      d->clearScreen();
      d->displayText("Ready to continue with the images");
      d->waitForKey();
      }*/

    //allow for a break after 50 trials then recalibrate

    if (trial==50)
      {
        d->displayText("You may take a break press space to continue when ready");
        d->waitForKey();
        // let's display an ISCAN calibration grid:
        d->clearScreen();
        d->displayISCANcalib();
        d->waitForKey();
        // let's do an eye tracker calibration:
        d ->displayText("<SPACE> to calibrate; other key to skip");
        int c = d->waitForKey();
        if (c == ' ') d->displayEyeTrackerCalibration(3, 3);
     }
    et->recalibrate(d,15);
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

extern "C" int main(const int argc, char** argv)
{
  // simple wrapper around submain() to catch exceptions (because we
  // want to allow PsychoDisplay to shut down cleanly; otherwise if we
  // abort while SDL is in fullscreen mode, the X server won't return
  // to its original resolution)
  try
    {
      return submain(argc, argv);
    }
  catch (...)
    {
      REPORT_CURRENT_EXCEPTION;
    }
  return 1;
}
// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
