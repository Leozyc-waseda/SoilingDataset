/*!@file AppPsycho/psycho-geon-relations.C Psychophysics display of geon images of Marks relations experiment */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-geon-relations.c $
// $Id: psycho-geon-relations.c 12962 2010-03-06 02:13:53Z irock $
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
#include <ctime>
#include <SDL/SDL_mixer.h> //for playing sounds
#include <stdio.h>

using namespace std;

char waitForResp(double WaitTime, double StartMeasure, *char keys, int NumOfKeys);
int WrightResults2File(string FileName,int* IsCorrect, double* RT,char* SubResp, char* CorrectResp, int NumOfTrials);


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

    int NumOfKeys=3;
    char keys[NumOfKeys]={'z', ',', '.'}; //from left to right, the key specified means object 1,2,3




  // Parse command-line:
  // Subject Run isAdapt(1or0) MaleOrFemale
  if (manager.parseCommandLine(argc, argv,
                               "<img1.ppm> ... <imgN.ppm>", 1, -1)==false){
        cout<<"please give the following arguments:\n Subject-Initials Run-Number  PresentationFrames(1 sec is 30 frames) MaleOrFemale(M for Male, F you guessed it...) Handedness(R L)"<< endl;
    cout<<"the keys for objects 1,2,3 are, respectively the keys:"<< keys[0]<< ", "<<keys[1] <<", "<< keys[2]<<endl;
        return(1);
        }

        string Sub=manager.getExtraArg(0);
        string Run=manager.getExtraArg(1);
        string PT=manager.getExtraArg(2);
        const char *pp;
        pp=PT.c_str();
        int PresentationFrames=atoi(pp);
        //int PresentationTime=(int)PT[0]-48;//atoi(PT[0].c_str());
        cout<<"presentation time is:"<<(PresentationFrames/30)<<" seconds"<<endl;
        string Gender=manager.getExtraArg(3);
        string Handedness=manager.getExtraArg(4);

        string DataDir="/lab/ilab19/geonexp/GRET/Data/";
        string psyFileName=DataDir + "GRET_Run" + Run + "_"+ Sub+"_StimONFrames-"+PT+"_"+Gender+"_"+Handedness+".psy";
        manager.setOptionValString(&OPT_EventLogFileName, psyFileName);
        manager.setOptionValString(&OPT_EyeTrackerType, "ISCAN");

        d->pushEvent(std::string("Run=") + Run + "\n" + Sub +"\nGender="+ Gender);




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

    string ExpDir="/lab/ilab19/geonexp/GRET/sequences/";
    string ImgDir="/lab/ilab19/geonexp/GRET/images/";


        FileName=ExpDir+"GeonRelationsET_Run"+Run+".txt";


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
  double RT[NumOfTrials]; //The reaction time for each trial
  char SubResp[NumOfTrials];//The actual response in each trial
  int IsCorrect[NumOfTrials]; // whether the response was correct (=1) incorrect (=0) or no response recorded (=-1)
  char  Correct_Response[NumOfTrials]; //The key that the subject should have pressed

  for(j=0; j<NumOfTrials; j++){
    getline(RunFile, ImageSequence[j]);
    ImageSequence[j]= ImgDir +ImageSequence[j];
    cout <<ImageSequence[j]<<endl;
  }

  RunFile.close();

  double StartMeasure;
  double Now;
  char Response;

  //setting up the audio (for the error-feedback)
  Mix_Music* recallCueMusic = NULL;
  //now let's open the audio channel
  if( Mix_OpenAudio( 22050, MIX_DEFAULT_FORMAT, 2, 4096 ) == -1 ){
       LINFO( "did not open the mix-audio") ;
       return -1 ;
       }
  SoundFile='/lab/ilab19/geonexp/GRET/images/beep-2.wav';
  BipSound = Mix_LoadMUS(SoundFile);





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

          //Finding out what object it is
          Current_Obj=ImageSequence[i][3];
          Correct_Response[i]=keys[ImageSequence[i][3]-1];

      // start the eye tracker:
      et->track(true);

      // blink the fixation:
      d->displayFixationBlink();

      // show the image:
      d->displaySurface(surf, -2);

      // Recording response:
          StartMeasure=d->getTimerValue();
          Response=waitForResp((PresentationFrames/30),  StartMeasure,  keys,  NumOfKeys); //see function bellow
          if (Response!='0'){RT[i]=((d->getTimerValue())-StartMeasure);};
          cout<<Response<<endl;

          //continues to present the image for the time specified
          Now = d->getTimerValue();
          while ((Now-StartMeasure)<(PresentationFrames/30)) { d->waitFrames(1);};

      // free the image:
      SDL_FreeSurface(surf);

      // make sure display if off before we stop the tracker:
      d->clearScreen();

      // stop the eye tracker:
      usleep(50000);
      et->track(false);

          //recording response if it wasn't yet recorded
          if (Response=='0') {Response=waitForResp(((PresentationFrames/30)+1),  StartMeasure,  keys,  NumOfKeys);}
          if (Response!='0'){RT[i]=((d->getTimerValue())-StartMeasure);}else{RT[i]=-1;};
          SubResp[i]=Response;


          //feedback if error
          if (Response!=Correct_Response){
                  IsCorrect[i]=0;
              if(task == 0){if( Mix_PlayMusic( BipSound, 0 ) == -1 ) { return 1; }
                                    while( Mix_PlayingMusic() == 1 ){}
                }else{
                IsCorrect[i]=1;
                }
        }
        string BehaveFileName=DataDir + "Behav_GRET_Run" + Run + "_"+ Sub+"_StimONFrames-"+PT+"_"+Gender+"_"+Handedness+".bhv";
        WrightResults2File(BehaveFileName,IsCorrect,  RT, SubResp,  Correct_Response, NumOfTrials);

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

// ######################################################################
char waitForResp(double WaitTime, double StartMeasure, *char keys, int NumOfKeys)
{
  // clear the input buffer before we get started if doWait is set to true
  while (checkForKey() != -1) ;

  double Now=StartMeasure;
  pushEventBegin("waitForResp");
  SDL_Event event;
  int Stop=0;

  do {

        Now = d->getTimerValue() ;
        SDL_PollEvent(&event);
        if (event.type == SDL_KEYDOWN){

                for (aa=0; aa<NumOfKeys; aa ++){ if  ((char(event.key.keysym.unicode))==keys[aa]) {Stop=1}}
        }

   } while (!Stop && (Now-StartMeasure)<WaitTime);

  if event.type==SDL_KEYDOWN{

        int key = event.key.keysym.unicode;
        char c; if (key >= 32 && key < 128) c = char(key); else c = '?';
        pushEventEnd(sformat("waitForResp - got %d (%c)", key, c));

        // abort if it was ESC:
        if (key == 27) LFATAL("Aborting on [ESC] key");

        return c;

  }else{  return '0';}

}

int WrightResults2File(string FileName,int* IsCorrect, double* RT,char* SubResp, char* CorrectResp, int NumOfTrials){
   /* function wrights the behavioral results of Mark's Geon Relations ET
     * experimnt into a text file */

    string ExpDir="/lab/ilab19/geonexp/GRET/Data/";
    FileName=ExpDir+FileName;
    const char *FileNameP=FileName.c_str();
    FILE * pFile;
    pFile = fopen (FileNameP,"w");




    int ii;

    for (ii=0; ii<NumOfTrials; ii ++){
        fprintf(pFile, "%i %f %c %c \n", IsCorrect[ii], RT[ii], SubResp[ii], CorrectResp[ii]);

    }

    fclose (pFile);

    return 0;






}
