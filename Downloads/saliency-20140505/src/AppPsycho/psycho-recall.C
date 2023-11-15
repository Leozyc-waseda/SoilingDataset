/*!@file AppPsycho/psycho-recall.C Test recall for objects in scenes */

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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-recall.C $
// $Id: psycho-recall.C 15024 2011-10-27 19:07:03Z dberg $
//

#include "Component/ModelManager.H"
#include "Component/EventLog.H"
#include "Image/Image.H"
#include "Image/CutPaste.H"
#include "Image/DrawOps.H"
#include "Image/ImageSet.H"
#include "Image/ShapeOps.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "Component/ComponentOpts.H"
#include "GUI/GUIOpts.H"
#include "Raster/Raster.H"
#include "Util/Types.H"
#include "Util/StringUtil.H"
#include "Util/StringConversions.H"
#include "Util/MathFunctions.H"
#include <fstream>


#define YES 118 //V key
#define NO  109 //M key
//! Test recall for objects in scenes
/*! This shows an image to be inspected under eye-tracking. Then a
  series of objects. For each object, the user makes a decision as to
  whether it was in the scene or not. Input file format should be:

  image_name upper_left_x upper_left_y size_x size_y obj_number
  ...

*/

// ######################################################################
// a simple structure to hold image info
// ######################################################################
struct ImageInfo
{
  struct obj
  {
    Point2D<int> ul;
    Dims size;
  };

  std::string name;
  std::vector<obj> objects;
};

// ######################################################################
static int submain(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Psychophysics Recall");

  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  nub::soft_ref<EyeTrackerConfigurator>
    etc(new EyeTrackerConfigurator(manager));
  manager.addSubComponent(etc);

  // Instantiate our various ModelComponents:
  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);

  // set a default display size:
  manager.exportOptions(MC_RECURSE);
  manager.setOptionValString(&OPT_SDLdisplayDims, "1920x1080");
  manager.setOptionValString(&OPT_EventLogFileName, "psychodata.psy");
  manager.setOptionValString(&OPT_EyeTrackerType, "EL");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<file_list> <distractor_list> <presentation_time> <min obj per trial> <max obj per trial> <average feedback> <images per query> <trials per break>", 3, -1) == false)
    return(1);

  // hook our various babies up and do post-command-line configs:
  nub::soft_ref<EyeTracker> et = etc->getET();
  d->setEyeTracker(et);
  d->setEventLog(el);
  et->setEventLog(el);
  initRandomNumbers();

  // let's get all our ModelComponent instances started:
  manager.start();

  const uint presentTime = fromStr<uint>(manager.getExtraArg(2));
  const uint minobjpertrial = fromStr<uint>(manager.getExtraArg(3));
  const uint maxobjpertrial = fromStr<uint>(manager.getExtraArg(4));
  const uint avgfeedback = fromStr<uint>(manager.getExtraArg(5));
  const uint imgsperquery = fromStr<uint>(manager.getExtraArg(6));
  const uint trialsPerBreak = fromStr<uint>(manager.getExtraArg(7));

  if(trialsPerBreak % imgsperquery != 0)
    LFATAL("The number of trials before a break must be divisible by the number of images per trial");

  // get the list of targets
  std::ifstream objectFile(manager.getExtraArg(0).c_str());
  if (objectFile == 0) LFATAL("Couldn't open file: '%s'",
                              manager.getExtraArg(0).c_str());

  // load up all the image descriptions:
  uint objcount = 0;
  std::vector<ImageInfo> targets;
  std::string line;
  uint cc = 0;
  while(getline(objectFile, line)) 
    {
      cc++;
      if (line[0] != '#')
        {
          std::vector<std::string> tokens;
          ImageInfo image;
          split(line, " ", std::back_inserter(tokens));
        
          if ((tokens.size() - 1) % 4 != 0)
            LFATAL("line %d, must have coordinates of upper left and size of each object", cc);
        
          image.name = tokens[0];
          for (uint ii = 1; ii < tokens.size(); ii+=4)
            {
              ImageInfo::obj object;
              uint x = fromStr<int>(tokens[ii]);
              uint y = fromStr<int>(tokens[ii+1]);
              object.ul = Point2D<int>(x,y);
            
              uint sx = fromStr<uint>(tokens[ii+2]);
              uint sy = fromStr<uint>(tokens[ii+3]);
              object.size = Dims(sx,sy);
            
              image.objects.push_back(object);
            }
          objcount += image.objects.size();
          targets.push_back(image);
        }
    }
  //close the objectFile
  objectFile.close();

  // get the list of distractor
  std::ifstream distFile(manager.getExtraArg(1).c_str());
  if (distFile == 0) LFATAL("Couldn't open file: '%s'",
                            manager.getExtraArg(1).c_str());

  // load up all the image descriptions:
  std::vector<std::vector<uint> > targetMap(targets.size());
  std::vector<ImageInfo> distractors;
  uint distcount = 0;
  uint distimgcount = 0;
  cc = 0;
  while(getline(distFile, line)) 
    {
      cc++;
      if (line[0] != '#')
        {
          std::vector<std::string> tokens;
          ImageInfo image;
          split(line, " ", std::back_inserter(tokens));
        
          if ((tokens.size() - 1) % 4 != 0)
            LFATAL("Line %d must have coordinates of upper left and size of each object", cc);
        
          image.name = tokens[0];
          for (uint ii = 1; ii < tokens.size(); ii+=4)
            {
              ImageInfo::obj object;
              uint x = fromStr<int>(tokens[ii]);
              uint y = fromStr<int>(tokens[ii+1]);
              object.ul = Point2D<int>(x,y);
            
              uint sx = fromStr<uint>(tokens[ii+2]);
              uint sy = fromStr<uint>(tokens[ii+3]);
              object.size = Dims(sx,sy);
            
              image.objects.push_back(object);
            }
          distcount += image.objects.size();
          distractors.push_back(image);
          ++distimgcount;
        }
      
      //every other line tells us which targets this distractor goes with
      getline(distFile, line);
      cc++;
      if (line[0] != '#')
        {
          std::vector<std::string> tokens;
          split(line, " ", std::back_inserter(tokens));
          for (uint ii = 0; ii < tokens.size(); ++ii)
            {
              const uint t = fromStr<uint>(tokens[ii]);
              if (t > targetMap.size())
                LFATAL("Distractor target map out of range(%d,%d)", t, (uint)targetMap.size());
              targetMap[t-1].push_back(distimgcount - 1);
            }
        }
    }
  //close the distFile
  distFile.close();

  //check to make sure every target has at least 1 distractor imag
  for (uint i = 0; i < targetMap.size(); ++i)
    if (targetMap[i].size() == 0)
      LFATAL("Each target must have at least 1 distractor image (none for %d)", i+1);
  
  // create a randomized index for the target images:
  uint targetidx[targets.size()];
  for (uint i = 0; i < targets.size(); i++) targetidx[i] = i; 
  randShuffle(targetidx, targets.size());
  
  //clear the screen and get going
  d->clearScreen();
  
  double runningAverage = 0.0;  
  uint totalObjectCount = 0;  
  
  //loop over all stimuli
  ImageSet< PixRGB<byte> > objects;  //hold the objects
  std::vector<int> groundTruth;      //hold ground truth (object number in text file)
  std::vector<std::string> objFileName; //hold distractor names
  for (uint ii = 0; ii < targets.size(); ++ii)
    {
      if (ii % trialsPerBreak == 0)
        {
          d->displayText("When you are ready place your chin on the rest, get comfortable, and press the space bar.");
          d->waitForKey();
          et->calibrate(d);
        }
    
      ImageInfo image = targets[targetidx[ii]];
      std::string name = image.name;
      std::vector<ImageInfo::obj> objectList = image.objects;
    
      // pick a random set of objects for this image
      uint posObjInd[objectList.size()];
      for (uint i = 0; i < objectList.size(); ++i) posObjInd[i] = i;
      randShuffle(posObjInd, objectList.size());
    
      //1/3 to 2/3 positives
      uint numobjects = minobjpertrial + randomUpToIncluding(maxobjpertrial - minobjpertrial);
      uint numfoils   = minobjpertrial + randomUpToIncluding(maxobjpertrial - minobjpertrial);
      
      // load up the targets and show a fixation cross on a blank screen:
      d->clearScreen();
    
      //load the target image
      LINFO("Loading '%s'...", name.c_str());

      if (numobjects > objectList.size())
        LFATAL("Cannot have more objects per trial than objects in the image.");

      Image< PixRGB<byte> > imt = Raster::ReadRGB(name);

      //get all the objects
      for (uint i = 0; i < numobjects; ++i)
        {
          LINFO("Loading object %u", i+1);
          if (!imt.rectangleOk(Rectangle(objectList[posObjInd[i]].ul, objectList[posObjInd[i]].size)))
            LFATAL("objects bounding box does not fit in the image");
      
          Image< PixRGB<byte> > obj = crop(imt, objectList[posObjInd[i]].ul, objectList[posObjInd[i]].size);
          objects.push_back(obj);
          groundTruth.push_back(posObjInd[i] + 1);
          objFileName.push_back(name);
        }

      //get all the distractors for this target
      std::vector<std::pair<ImageInfo::obj, std::string> > distList;
      std::vector<uint> distObjPosList;
      std::vector<uint> distpos = targetMap[targetidx[ii]];

      for (uint i = 0; i < distpos.size(); ++i)
        {
          ImageInfo distractor = distractors[distpos[i]];
          std::string dname = distractor.name;
          std::vector<ImageInfo::obj> dlist = distractor.objects;
          for (uint j = 0; j < dlist.size(); ++j)
            {
              distList.push_back(std::make_pair(dlist[j], dname));
              distObjPosList.push_back(j);
            }
        }

      // pick a random set of distractors
      uint posDistInd[distList.size()];
      for (uint i = 0; i < distList.size(); ++i) posDistInd[i] = i;
      randShuffle(posDistInd, distList.size());
      
      if (numfoils > distList.size())
        LFATAL("Requesting more foils than distractors for this target");

      //load the foil objects
      for (uint i = 0; i < numfoils; ++i)
        {
          std::pair<ImageInfo::obj, std::string> p = distList[posDistInd[i]];
          LINFO("Loading '%s'...", p.second.c_str());
          Image< PixRGB<byte> > imtd = Raster::ReadRGB(p.second);
          LINFO("Loading distractor %u", i+1);

          if (!imtd.rectangleOk(Rectangle(p.first.ul, p.first.size)))
            LFATAL("objects bounding box does not fit in the image");
      
          Image< PixRGB<byte> > obj = crop(imtd, p.first.ul, p.first.size);
          objects.push_back(obj);
          groundTruth.push_back(-1 * (distObjPosList[posDistInd[i]]+1));
          objFileName.push_back(p.second);
        }
    
      // Prepare to display the image, and a random mask:
      SDL_Surface *surf = d->makeBlittableSurface(imt, true);
    
      Image< PixRGB<byte> > mask(d->getWidth()/8, d->getHeight()/8, NO_INIT);
      for (int i = 0; i < mask.getSize(); i ++)
        mask.setVal(i, PixRGB<byte>(byte(randomUpToNotIncluding(256)),
                                    byte(randomUpToNotIncluding(256)),
                                    byte(randomUpToNotIncluding(256))));
      mask = zoomXY(mask, 8, 8);
      SDL_Surface *msurf = d->makeBlittableSurface(mask, true);
      LINFO("%s ready.", name.c_str());
      
      // ready to go whenever the user is ready:
      d->displayFixation();
      d->waitForKey();
      d->pushEvent(std::string("===== Showing image: ") +
                   name + " =====");
      
      // start the eye tracker:
      et->track(true);
      
      // show the image:
      d->displaySurface(surf, -2);
      
      // fixed presentation time:
      usleep(presentTime);
      
      // Display mask:
      d->displaySurface(msurf, -2);
      usleep(150000);
      
      // make sure display if off before we stop the tracker:
      d->clearScreen();
      
      // stop the eye tracker:
      usleep(50000);
      et->track(false);
      SDL_FreeSurface(surf);
      SDL_FreeSurface(msurf);
    
      if ((ii+1) % imgsperquery != 0)
        {
          d->clearScreen();
          continue;
        }
    
      //randomly order the objects
      uint objidx[objects.size()];
      for (uint i = 0; i < objects.size(); ++i) objidx[i] = i;
      randShuffle(objidx, objects.size());
      
      // Allow the user to mark some objects:
      for (uint i = 0; i < objects.size(); ++i)
        {
          // create an image with the object:
          Image< PixRGB<byte> > obj(d->getDims(), NO_INIT);
          obj.clear(PixRGB<byte>(64,64,64));

          Point2D<int> p((d->getWidth() - objects[objidx[i]].getWidth())/2,
                         (d->getHeight() - objects[objidx[i]].getHeight())/2);

          LINFO("Displaying object %u from %s, at %u,%u, with size %ux%u", i, objFileName[objidx[i]].c_str(), p.i, p.j, 
                  objects[objidx[i]].getWidth(), objects[objidx[i]].getHeight());

          if (objects[objidx[i]].size() > obj.size())
            LFATAL("Object must be small than the display");

          inplacePaste(obj, objects[objidx[i]], p);
          // show the object:
          SDL_Surface *surf2 = d->makeBlittableSurface(obj, true);
          d->displaySurface(surf2, -2);

          //get their response and make sure its a 1 or 0
          int c = d->waitForKey();
          while ((c != YES) && (c != NO))
            c = d->waitForKey();

          int gt = groundTruth[objidx[i]];
          bool isCorrect = (((c == NO) && (gt < 0)) || ((c == YES) && (gt >= 0)));
          
          if (gt > 0)
            d->pushEvent(sformat("===== Object Target %s - %d: '%c' (%d) : %u =====",
                                 objFileName[objidx[i]].c_str(), gt, c, c, isCorrect));
          else if (gt < 0)
            d->pushEvent(sformat("===== Object Foil %s - %d: '%c' (%d) : %u =====",
                                 objFileName[objidx[i]].c_str(), gt, c, c, isCorrect));
          else
            LFATAL("ground truth cannot be 0");


          
          SDL_FreeSurface(surf2);
          
          //update the running average
          ++totalObjectCount;
          runningAverage = (isCorrect + (totalObjectCount - 1) * runningAverage) / totalObjectCount;
        }

      objects = ImageSet<PixRGB<byte> >();
      groundTruth.clear();
      objFileName.clear();
 
      //should we report their running average
      if (randomUpToNotIncluding(avgfeedback) == 0)
        {
          d->clearScreen();
          
          if (runningAverage < 0.5)
            d->displayText(sformat("%3.2f percent correct\nYou can do better!",runningAverage * 100));
          else if (runningAverage < 0.75)
            d->displayText(sformat("%3.2f percent correct\nKeep it up!",runningAverage * 100));
          else 
            d->displayText(sformat("%3.2f percent correct\nGood job! ",runningAverage * 100));
          
          d->waitForKey();
        }
      
      // clear the screen for the next trial
      d->clearScreen();
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
