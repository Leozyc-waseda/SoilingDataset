/*!@file AppPsycho/psycho-geon.C Psychophysics display of geon images */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-geon.C $
// $Id: psycho-geon.C 12962 2010-03-06 02:13:53Z irock $
//

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "Raster/Raster.H"
#include "Util/MathFunctions.H"
#include "Util/Types.H"
#include <fstream>

using namespace std;

// ######################################################################
extern "C" int main(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Psycho Still");

  // Instantiate our various ModelComponents:
  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  nub::soft_ref<EyeTrackerConfigurator>
    etc(new EyeTrackerConfigurator(manager));
  manager.addSubComponent(etc);

  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);




  // Parse command-line:
  // Subject Run isAdapt(1or0) MaleOrFemale
  if (manager.parseCommandLine(argc, argv,
                               "<img1.ppm> ... <imgN.ppm>", 1, -1)==false){
        cout<<"please give the following arguments:\n Subject-Initials Run-Number Is-Adaptation(1 for adaptation 0 for control) PresentationTime(1 or 2 in secs) MaleOrFemale(M for Male, F you guessed it...) Handedness(R L)"<< endl;
    return(1);
        }


        string Sub=manager.getExtraArg(0);
        string Run=manager.getExtraArg(1);
        string IA=manager.getExtraArg(2);
        int IsAdapt=(int)IA[0]-48;//
        cout<<IsAdapt<<"is adaptation"<<endl;
        string PT=manager.getExtraArg(3);
        int PresentationTime=(int)PT[0]-48;//atoi(PT[0].c_str());
        cout<<"presentation time is:"<<PresentationTime<<endl;
        string Gender=manager.getExtraArg(4);
        string Handedness=manager.getExtraArg(5);

        string DataDir="/lab/ilab19/geonexp/stimuli/Data/";
        string psyFileName=DataDir + "BG_Run" + Run + "_Adapt" + IA +"_"+ Sub+"_StimONTime-"+PT+"_"+Gender+"_"+Handedness;
        manager.setOptionValString(&OPT_EventLogFileName, psyFileName);
        manager.setOptionValString(&OPT_EyeTrackerType, "ISCAN");

        d->pushEvent(std::string("Run=") + Run + "\n" + Sub +"\nGender="+ Gender);
        if (IsAdapt) {d->pushEvent("Adaptation");}
        else {d->pushEvent("Control");};



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
  d->displayText("<SPACE> for random play; other key for ordered");
  //int c = d->waitForKey();

  // setup array of movie indices:
  //uint nbimgs = manager.numExtraArgs(); int index[nbimgs];
  //for (uint i = 0; i < nbimgs; i ++) index[i] = i;
  //if (c == ' ') { LINFO("Randomizing images..."); randShuffle(index, nbimgs); }

  //Reads the Images sequence text file:
    string FileName;

    string ExpDir="/lab/ilab19/geonexp/stimuli/sequences/";
    string ImgDir="/lab/ilab19/geonexp/stimuli/images/";

    if (IsAdapt){
        FileName=ExpDir+"GRun_"+Run+"Adapt.txt";
    }else{
        FileName=ExpDir+"GRun_"+Run+"Control.txt";
    };

    cout<<FileName<<endl;
    const char *FileNameP=FileName.c_str();


  ifstream RunFile (FileNameP);
  string  u;

  int NumOfTrials, j;

  getline(RunFile, u);
  cout<<u;
  const char *p;
  p=u.c_str();
  NumOfTrials=atoi(p);
  string ImageSequence[NumOfTrials];

  for(j=0; j<NumOfTrials; j++){
    getline(RunFile, ImageSequence[j]);
    ImageSequence[j]= ImgDir +ImageSequence[j];
    cout <<ImageSequence[j]<<endl;
  }

  RunFile.close();


  LINFO("Ready to go");

  // main loop:
  for (int i = 0; i < NumOfTrials; i ++)
    {
      // load up the frame and show a fixation cross on a blank screen:
      d->clearScreen();
      LINFO("Loading '%s'...", ImageSequence[i].c_str());
      Image< PixRGB<byte> > image =
        Raster::ReadRGB(ImageSequence[i]);

      SDL_Surface *surf = d->makeBlittableSurface(image, true);

      LINFO("'%s' ready.", ImageSequence[i].c_str());
      d->displayFixation();

      // ready to go whenever the user is ready:
      d->waitForKey();
      d->waitNextRequestedVsync(false, true);
      d->pushEvent(std::string("===== Showing image: ") +
                   ImageSequence[i] + " =====");

      // start the eye tracker:
      et->track(true);

      // blink the fixation:
      d->displayFixationBlink();

      // show the image:
      d->displaySurface(surf, -2);

      // wait for key:
      d->waitFrames(PresentationTime*30);

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
