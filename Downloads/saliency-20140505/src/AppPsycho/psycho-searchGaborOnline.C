/*!@file AppPsycho/psycho-searchGaborOnline.C Psychophysics display for a search for a
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
// Primary maintainer for this file: Farhan Baluch <fbaluch@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-searchGaborOnline.C $
// $Id: psycho-searchGaborOnline.C 10794 2009-02-08 06:21:09Z itti $
//

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
#include "Util/stats.H"
#include "Util/Timer.H"
#include "Image/ShapeOps.H"
#include "Image/DrawOps.H"
#include <ctype.h>
#include <vector>
#include <string>
#include <fstream>
#include <deque>

using namespace std;

//! number of frames in the mask
#define NMASK 10

struct gPatch
{
    int patchNo;
    Point2D<int> pos;
};


// ######################################################################
static int submain(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Psycho Search");

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
  if (manager.parseCommandLine(argc, argv, "<imagelist.txt> <subjectname>", 2, 2) == false)
    return(1);

  // hook our various babies up and do post-command-line configs:
  nub::soft_ref<EyeTracker> et = etc->getET();
  d->setEyeTracker(et);
  d->setEventLog(el);
  et->setEventLog(el);
  //ONLINE
  et->requestQuickEyeS();

  // let's get all our ModelComponent instances started:
  manager.start();
  string subjName =std::string(manager.getExtraArg(1).c_str());

  // let's pre-load all the image names so that we can randomize them later:
  FILE *f = fopen(manager.getExtraArg(0).c_str(), "r");
  if (f == NULL) LFATAL("Cannot read stimulus file");
  char line[1024];
  std::vector<std::string> ilist, tlist, rlist, slist;
int correct = 0, accuracy = 100, total = 0;

  while(fgets(line, 1024, f))
    {
       std::vector<std::string> tokens;
     // each line has four filenames: first the imagelet that contains
      // only the target, second the image that contains the target in
      // its environment, third the image to report position of target
      //fourth the name of the spec file for the search array
      LINFO("line reads %s",line);
      split(line," ", std::back_inserter(tokens));

      // now line is at the imagelet, line2 at the image
      tlist.push_back(std::string(tokens[0]));
      ilist.push_back(std::string(tokens[1]));
      rlist.push_back(std::string(tokens[2]));
      slist.push_back(std::string(tokens[3]));
      LINFO("\nNew pair \nline1 reads: %s, \nline2 reads:%s, \nline 3 reads %s,\nline 4 reads %s,",tokens[0].c_str(), tokens[1].c_str(),tokens[2].c_str(), tokens[3].c_str());

    }
  fclose(f);

  // randomize stimulus presentation order:
  int nimg = ilist.size(); int imindex[nimg];
  for (int i = 0; i < nimg; i ++) imindex[i] = i;
  randShuffle(imindex, nimg);

  // let's display an ISCAN calibration grid:
  d->clearScreen();
  d->displayISCANcalib();
  d->waitForKey();

  // let's do an eye tracker calibration:
  d->displayText("<SPACE> to calibrate; other key to skip");
  int c = d->waitForKey();
  if (c == ' ') et->calibrateOnline(d);

  // we are ready to start:
  d->clearScreen();

  d->displayText("<SPACE> to start experiment");
  d->waitForKey();

//store the latest 120 samples~500ms worth of data
  int sampleCnt=0,numSamples=50;
  std::deque<Point2D<int> > currentEyeSamples(numSamples);
  Point2D<int>meanFixation;
  stats<float> Stats;


  //******************* main loop:***********************//
  for (int im = 0; im < nimg; im++) {
    int imnum = imindex[im];

    // load up the images and show a fixation cross on a blank screen:
    d->clearScreen();
    LINFO("Loading '%s' / '%s'...", ilist[imnum].c_str(),tlist[imnum].c_str());

    // get the imagelet and place it at a random position:

    if(!Raster::fileExists(tlist[imnum]))
    {
      // stop all our ModelComponents
      manager.stop();
      LFATAL("i couldnt find image file %s", tlist[imnum].c_str());
    }

    Image< PixRGB<byte> > targetImg = Raster::ReadRGB(tlist[imnum]);
    Image< PixRGB<byte> > rndimg(d->getDims(), NO_INIT);
    rndimg.clear(d->getGrey());
    int rndx = 0,rndy = 0;

    SDL_Surface *surf1 = d->makeBlittableSurface(targetImg, true);
    char buf[256];

    if(!Raster::fileExists(ilist[imnum]))
    {
      manager.stop();
      LFATAL("i couldnt find image file %s", ilist[imnum].c_str());
    }
    Image<PixRGB<byte> > ArrayImg = Raster::ReadRGB(ilist[imnum]);
    SDL_Surface *surf2 = d->makeBlittableSurface(ArrayImg, true);

    //randShuffle(mindex, NMASK);

   // load up the reporting number image:
       if(!Raster::fileExists(rlist[imnum]))
    {
      // stop all our ModelComponents
      manager.stop();
      LFATAL(" i couldnt find image file %s", tlist[imnum].c_str());
    }

    Image< PixRGB<byte> > reportImg = Raster::ReadRGB(rlist[imnum]);
    //SDL_Surface *surf3 = d->makeBlittableSurface(reportImg, true);

    // give a chance to other processes if single-CPU:
    usleep(200000);

    // ready to go whenever the user is ready:
    d->displayFixationBlink();
    //d->waitForKey();
    d->waitNextRequestedVsync(false, true);

    //************************ Display target************************//
    sprintf(buf, "===== Showing imagelet: %s at (%d, %d) =====",
            tlist[imnum].c_str(), rndx, rndy);

    d->pushEvent(buf);
    d->displaySurface(surf1, 0, true);
    usleep(2000000);

    d->clearScreen();
    SDL_FreeSurface(surf1);
    //************************ Display array************************//

     // start the eye tracker:
    et->track(true);

    std::vector<std::string> stimNameTokens;
    split(ilist[imnum],"/",std::back_inserter(stimNameTokens));

    et->setCurrentStimFile(std::string(subjName + "_" + stimNameTokens[stimNameTokens.size()-1]));
    currentEyeSamples.resize(0);
    sampleCnt =0;

    // show the image:
      d->pushEvent(std::string("===== Showing search image: ") + ilist[imnum] +
                 std::string(" ====="));


    d->displaySurface(surf2, 0, true);

    Image<PixRGB<byte> > dispImage(1920,1080,NO_INIT);
    // wait for key; it will record reaction time in the logs:
    d->checkForKey();
    Timer timer(100);
    timer.reset();
    double startTime=timer.getSecs();
    bool timeOut=false;
    while(d->checkForKey() < 0 && !timeOut)
    {
      dispImage = ArrayImg;
      //ONLINE
      Point2D<int> tempEye = et->getCalibEyePos();
      if(tempEye.i >0 && tempEye.i <1960 && tempEye.j>0 &&tempEye.j<1080)
        currentEyeSamples.push_back(tempEye);
      // Point2D<int> tempEye(960,540);
      // currentEyeSamples.push_back(tempEye);
      //drawCircle(dispImage, Point2D<int> (currentEyeSamples.back()),5,PixRGB<byte>(255,255,255));
      drawCircle(dispImage, tempEye,5,PixRGB<byte>(255,255,255));
      surf2 = d->makeBlittableSurface(dispImage,true);
      d->displaySurface(surf2,-2);
      LINFO("getting live eyepos %d,%d",currentEyeSamples.back().i,
              currentEyeSamples.back().j);

      if(currentEyeSamples.size() ==(size_t)numSamples)
        {
          currentEyeSamples.pop_front();
          sampleCnt=numSamples;
        }
      if(sampleCnt < numSamples);
       sampleCnt++;

      SDL_FreeSurface(surf2);
      if(timer.getSecs() >= startTime + 10)
        timeOut=true;
    }


    // stop the eye tracker:
    et->track(false); //we just want to record eye movements while subjects view the array
    LINFO("getting out of track mode");
    LINFO("currenteye samples available %d",(int)currentEyeSamples.size());
    //lets take the mean of the last 120 samples and assume that is the target
    //d::deque<Point2D<int> >::const_iterator itr = currentEyeSamples.begin();
    std::vector<float> xVals(sampleCnt),yVals(sampleCnt);
    float xMean,yMean;

    Point2D<int>temp;
    for(int i =0;i <(int)currentEyeSamples.size();i++) //sampleCnt);
    {
      LINFO("accessing front cnt=%d %d",i,currentEyeSamples[i].i);
      temp = currentEyeSamples[i];
      LINFO("\n counting %d,%d",temp.i,temp.j);
        xVals[i]=temp.i;
        yVals[i]=temp.j;
    }

    if(sampleCnt > 0)
      {
        xMean = Stats.mean(xVals);
        yMean = Stats.mean(yVals);
      }
    else
      {
        xMean = 0;
        yMean=0;
      }

    LINFO("mean target %f,%f",xMean,yMean);
    //check user response
    char tmp[40];

    //lets open the spec file and extract the target number
    ifstream specFile(slist[imnum].c_str(), ifstream::in);

    bool found =false;
    string testLine;
    std::vector<std::string> specTokens,xTokens,yTokens,gTokens;
    std::vector<gPatch> gPatches(32);
    int targetNum=0,patchCnt=0;

    LINFO("opening specfile");
   if(specFile.is_open())
      {
        while(!specFile.eof() && !found)
          {
            getline(specFile, testLine);
            LINFO("got line %s",testLine.c_str());
            string::size_type loc = testLine.find("target", 0);
            string::size_type loc2 = testLine.find("Gabor",0);
            string::size_type loc3 = testLine.find("Xpos",0);
            string::size_type loc4 = testLine.find("Ypos",0);

            if(loc != string::npos)
            {
                split(testLine," ", std::back_inserter(specTokens));
                targetNum = atoi(specTokens[1].c_str());
                specTokens.resize(0);
                found = true;
            }

            if(loc2 != string::npos)
            {

                    split(testLine," ", std::back_inserter(gTokens));
                    gPatches[patchCnt].patchNo = atoi(gTokens[1].c_str());
                    LINFO("gPatches[%d].patchNo = %d",patchCnt,gPatches[patchCnt].patchNo);
                    gTokens.resize(0);

            }

            if(loc3 != string::npos)
            {
                split(testLine,"\t", std::back_inserter(xTokens));
                gPatches[patchCnt].pos.i = atoi(xTokens[1].c_str())+120;
                LINFO("gPatches[%d].pos.i = %d",patchCnt,gPatches[patchCnt].pos.i);
                xTokens.resize(0);
            }
            if(loc4 != string::npos)
            {
                split(testLine,"\t", std::back_inserter(yTokens));
                gPatches[patchCnt].pos.j = atoi(yTokens[1].c_str())+120;
                LINFO("gPatches[%d].pos.j = %d",patchCnt,gPatches[patchCnt].pos.j);
                yTokens.resize(0);
                if(patchCnt < 31)
                      patchCnt++;
             }

           }

        LINFO("specfile parsed finding target x y");

         if(!found)
          {
            manager.stop();
            LFATAL("couldnt find the target number from spec file");
          }


       //now lets find which patch they were looking at and check if
       //that was a target
        std::vector<float> euclidDists(32);
        float minDist=500000;
        int minIdx=-1;


        for(int i=0;i<32;i++)
        {
          LINFO("finding distance between fix %f,%f and gabor%d,%d",xMean,yMean,gPatches[i].pos.i,gPatches[i].pos.j);
            euclidDists[i] = gPatches[i].pos.distance(Point2D<int>((int)xMean,(int)yMean));
            if(euclidDists[i] < minDist)
            {
                minDist = euclidDists[i];
                minIdx = i;
            }
            LINFO("euclidDist %d %f",i,euclidDists[i]);

        }
        if(minIdx == -1)
          {  minIdx =0;
            LINFO("cannot find a match in euclid distance");
          }
        LINFO("found min IDx to be %d", minIdx);

        int txPos = gPatches[minIdx].pos.i, tyPos = gPatches[minIdx].pos.j;
        SDL_Surface *surf4;
        //while(d->checkForKey() < 0)
        // {
        for(int i=0;i<20;i++)
          {

            // int txPos = gPatches[i].pos.i, tyPos = gPatches[i].pos.j;
            //    LINFO("txpos typos (%d,%d)",txPos,tyPos);
            drawCircle(ArrayImg, Point2D<int> (txPos,tyPos),5,PixRGB<byte>(255,255,255));
            surf4 = d->makeBlittableSurface(ArrayImg, true);
            d->displaySurface(surf4,-2);
            drawCircle(ArrayImg, Point2D<int> (txPos,tyPos),5,PixRGB<byte>(255,0,0));
            //drawRect(ArrayImg,Rectangle::tlbrI(tyPos-(240/2),txPos - (240/2),tyPos+(240/2), txPos+(240/2)),PixRGB<byte>(0,0,0),5);
            surf4 = d->makeBlittableSurface(ArrayImg, true);
            d->displaySurface(surf4,-2);
          }


             SDL_FreeSurface(surf4);
            // }

       LINFO("after showing selection");

        total++;
        if (minIdx == targetNum)
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

   LINFO("out of spec file zone");
   // free the imagelet and image:
   //SDL_FreeSurface(surf1); SDL_FreeSurface(surf2); //SDL_FreeSurface(surf3);
   //SDL_FreeSurface(surf4);


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

       if (im==50)
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
        if (c == ' ') et->calibrateOnline(d);
        }
  }


  //et->recalibrate(d,20);
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
