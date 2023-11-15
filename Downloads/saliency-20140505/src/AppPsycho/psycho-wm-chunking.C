/*!@file AppPsycho/psycho-wm-chunking.C Psychophysics test to measure the influence of eyemovement on memory task performance */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-wm-chunking.C $


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
#include "GameBoard/basic-graphics.H"
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>
#include <iostream>
#include <SDL/SDL.h>
#include <SDL/SDL_image.h>
#include <SDL/SDL_mixer.h>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <time.h>
#include "Image/DrawOps.H"
#include "GameBoard/resize.h"
#include <iostream>
#include <fstream>
#include <set>
#include <algorithm>
#include <ctime>

#ifndef INVT_HAVE_LIBSDL_IMAGE
#include <cstdio>
int main()
{
        fprintf(stderr, "The SDL_image library must be installed to use this program\n");
        return 1;
}

#else



using namespace std;

// ######################################################################

ModelManager manager("psycho-wm-chunking");
nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
map<uint,uint> testMap ;
map<string,string> argMap ;
map<string,vector<SDL_Rect*>*> clipsmap;

//////////////////////////////////////////////
// a functionf for stringigying things
//////////////////////////////////////////////
template <class T> std::string stringify(T i)
{
        ostringstream o ;
        o << i ;
        return o.str();
}

bool itIsInThere(int x , vector<int> bag){
        for( uint i=0 ; i < bag.size(); i++ ){
                if(x == bag[i]) return true ;
        }
        return false ;
}



////////////////////////////////////////////////////////
///////this will change the order of elements in a vector to a random order
////////////////////////////////////////////////////////
void scramble(vector<int>& v){
        vector<int> tv = vector<int>() ;
        while(v.size()>0){
                tv.push_back(v[0]);
                v.erase(v.begin());
        }
        int i = 0 ;
        while(tv.size()>0){
                i = rand()%tv.size() ;
                v.push_back(tv[i]);
                tv.erase(tv.begin()+i);
        }
}



//pushes back the name of wav files in the directory into the given vector
int getdir (string dir, vector<string> &files)
{
        DIR *dp;
        struct dirent *dirp;
        if((dp  = opendir(dir.c_str())) == NULL) {
                cout << "Error(" << errno << ") opening " << dir << endl;
                return errno;
        }
        string fn = "" ;
        size_t found;
        string extension = "" ;
        while ((dirp = readdir(dp)) != NULL) {
                fn = string(dirp->d_name) ;
                found = fn.find_last_of(".");
                if(found > 0 && found <1000){
                        extension = fn.substr(found) ;
                        if(extension.compare(".wav")== 0 )
                                files.push_back(dir+"/"+fn);
                }
        }
        closedir(dp);
        return 0;
}








int addArgument(const string st,const string delim="="){
        int i = st.find(delim) ;
        argMap[st.substr(0,i)] = st.substr(i+1);

        return 0 ;
}

std::string getArgumentValue(string arg){
        return argMap[arg] ;
}

std::vector<int> getDigits(int n , string zs="n" , string rs="n" ){
        if(rs.compare("n")==0){
                if(zs.compare("n")==0 && n >9 ) {LINFO( "come on! what do you expect?!") ;  exit(-1) ;}
                if(zs.compare("y")==0 && n >10 ) {LINFO( "come on! what do you expect?!") ; exit(-1) ;}
        }
        vector<int> digits ;
        int dig = 0 ;
        while( digits.size() < (uint)n ){
                if(zs.compare("n")==0) {dig = 1+(random()%9);}else{dig = random()%10;}
                if(rs.compare("y")==0){digits.push_back(dig);}else{if(!itIsInThere(dig,digits)) digits.push_back(dig);}
        }
        return digits ;
}

std::string getUsageComment(){

        string com = string("\nlist of arguments : \n");

        com += "\nlogfile=[logfilename.psy] {default = psycho-wm-chunking.psy}\n" ;
        com += "\nmemo=[a_string_without_white_space]\n";
        com += "\nnum-of-digits=[>0](the size of string){default=3} \n";
        com += "\nsubject=[subject_name] \n" ;
        com += "\nseperate-recall-rounds=[>1] (number of trials including seperate recall task ) {default=2}\n";
        com += "\nfirst-to-last-rounds=[>1](number of trials including first to last chunking){default=2}}\n";
        com += "\nlast-to-first-rounds=[>1](number of trials including last to firts chunking){default=2}}\n";
        com += "\nalphabet=[a string of characters](a string of characters){default=0123456789}\n";
        com += "\ncue-wait-frames=[>0](number of frames to show the cue){default=0}\n";
        com += "\ncue-onset-frames=[>0](number of frames to show the cue onset){default=10}\n";
        com += "\nsound-dir=[path to wav files directory]{default=..}\n";
        com += "\ndigit-repeat=[y/n](whether digits can repeat, y for yes , n for no){default=n}\n" ;
        com += "\ninclude-zero=[y/n](whether zero be included in the presented digits, y for yes , n for no){default=n}\n";
        com += "\ndelay=[>1] (delay in number of frames in which recording happens)\n" ;
        return com ;
}


extern "C" int main(const int argc, char** argv)
{

        MYLOGVERB = LOG_INFO;  // suppress debug messages
        //let's push the initial value for the parameters
        argMap["experiment"]="chunking firs to last, chunking last to first and no chunking";
        argMap["logfile"]="psycho-wm-chunking.psy" ;
        argMap["num-of-digits"]="3" ;
        argMap["seperate-recall-rounds"]="2";
        argMap["first-to-last-rounds"] = "2" ;
        argMap["last-to-first-rounds"] = "2" ;
        argMap["subject"]="" ;
        argMap["memo"]="" ;
        argMap["alphabet"]="0123456789";
        argMap["cue-wait-frames"]="0" ;
        argMap["cue-onset-frames"] = "10" ;
        argMap["sound-dir"]="..";
        argMap["include-zero"]="n";
        argMap["digit-repeat"]="n" ;
        argMap["delay"] = "120" ;

        manager.addSubComponent(d);
        nub::soft_ref<EventLog> el(new EventLog(manager));
        manager.addSubComponent(el);
        d->setEventLog(el);
        nub::soft_ref<EyeTrackerConfigurator>
                        etc(new EyeTrackerConfigurator(manager));
          manager.addSubComponent(etc);

        if (manager.parseCommandLine(argc, argv,
            "at least one argument needed", 1, -1)==false){
                    cout<<getUsageComment()<<endl;
                    return(1);
            }

        for(uint i = 0 ; i < manager.numExtraArgs() ; i++){
                    addArgument(manager.getExtraArg(i),std::string("=")) ;
        }

        manager.setOptionValString(&OPT_EventLogFileName, argMap["logfile"]);
        manager.setOptionValString(&OPT_EyeTrackerType, "ISCAN");
        nub::soft_ref<EyeTracker> eyet = etc->getET();
        d->setEyeTracker(eyet);
        eyet->setEventLog(el);

        //now let's open the audio channel
         if( Mix_OpenAudio( 22050, MIX_DEFAULT_FORMAT, 2, 4096 ) == -1 ){
                    LINFO( "did not open the mix-audio") ;
                    return -1 ;
            }
        //let's load the 10 audio files off the director of audios and put them in a map
        map<int,Mix_Music*> audioMap ;
        for( int i = 0 ; i < 10 ; i++ ){
                string str = argMap["sound-dir"]+"/"+stringify(i)+".wav" ;
                audioMap[i] = Mix_LoadMUS(str.c_str());
        }
        int delay = atoi(argMap["delay"].c_str()) ;
        int numOfDigits = atoi(argMap["num-of-digits"].c_str());
        int cue_onset_frames = atoi(argMap["cue-onset-frames"].c_str()) ;
        int cue_wait_frames = atoi(argMap["cue-wait-frames"].c_str()) ;
        int num_of_first_task = atoi(argMap["seperate-recall-rounds"].c_str());
        int num_of_second_task = atoi(argMap["first-to-last-rounds"].c_str());
        int num_of_third_task = atoi(argMap["last-to-first-rounds"].c_str());
        vector<int>* taskVector = new vector<int>();
        for(int i = 0 ; i < num_of_first_task ; i++)  taskVector->push_back(1);
        for(int i = 0 ; i < num_of_second_task ; i++)  taskVector->push_back(2);
        for(int i = 0 ; i < num_of_third_task ; i++)  taskVector->push_back(3);
        scramble(*taskVector);

  // let's get all our ModelComponent instances started:
            manager.start();
            for(map<string,string>::iterator it= argMap.begin(); it!= argMap.end() ; ++it) d->pushEvent("arg:"+ it->first+" value:"+it->second ) ;
  // let's display an ISCAN calibration grid:
            d->clearScreen();
            d->displayISCANcalib();
            d->waitForMouseClick();
            d->displayText("Here the experiment starts! click to start!");
            d->waitForMouseClick();
            d->clearScreen();
            //let's do calibration
            d->displayText("CLICK LEFT button to calibrate; RIGHT to skip");
            int cl = d->waitForMouseClick();
            if (cl == 1) d->displayEyeTrackerCalibration(3,5,1 , true);
            d->clearScreen();

            Uint32 fc = d->getUint32color(PixRGB<byte>(255,0,0));
            Uint32 sc = d->getUint32color(PixRGB<byte>(0,255,0));
            Uint32 tc = d->getUint32color(PixRGB<byte>(0,0,255));
            SDL_Surface* f_pad= getABlankSurface(36,34);
            SDL_Surface* s_pad= getABlankSurface(36,34);
            SDL_Surface* t_pad= getABlankSurface(36,34) ;
            fillRectangle(f_pad,fc,0,0,35 ,33);
            fillRectangle(s_pad,sc,0,0,35 ,33);
            fillRectangle(t_pad,tc,0,0,35 ,33);
            SDL_Rect cue_offset ;
            cue_offset.x = (d->getWidth() -36) /2;
            cue_offset.y = (d-> getHeight() - 34) /2;
        int fs = 0 ;
        int ss = 0;
        int ts = 0 ;
            for( uint r = 0 ; r < taskVector->size() ; r++ ){
                d->showCursor(true);
                d->displayText("click one of the  mouse buttons to start!");
                d->waitForMouseClick() ;
                d->showCursor(false) ;
                d->clearScreen() ;
                d->displayFixationBlink();
                int task = taskVector->at(r) ;
                vector<int> digits = getDigits(numOfDigits,argMap["include-zero"],argMap["digit-repeat"]);
                    int counter = 0 ;
                    while( counter < numOfDigits ){
                        if(Mix_PlayingMusic() == 0 ){
                                //Play the music
                                if( Mix_PlayMusic( audioMap[digits[counter]], 0 ) == -1 ) { return 1; }
                                d->pushEvent("the "+stringify(counter)+"th  : " +stringify(digits[counter]));
                                counter++ ;
                        }
                    }
                //just hold it there to the end of the audio playing
                while( Mix_PlayingMusic() == 1 ){}
                if(cue_wait_frames != 0) d->waitFrames(cue_wait_frames) ;
                std::string imst = "===== Showing image: def";
                PixRGB<byte> cuecolor;
                switch( task ){
                        case 1 : fs++ ; d->displaySDLSurfacePatch(f_pad , &cue_offset,NULL , -2,false, true); imst += "_first_"+stringify(fs)+".png" ; break ;
                        case 2 : ss++ ; d->displaySDLSurfacePatch(s_pad , &cue_offset,NULL , -2,false, true);imst += "_second_"+stringify(ss)+".png" ;break ;
                        case 3 : ts++ ; d->displaySDLSurfacePatch(t_pad , &cue_offset,NULL , -2,false, true);imst += "_third_" +stringify(ts)+".png";break ;
                }

                d->pushEvent(imst);
                d->waitFrames(cue_onset_frames);
                eyet->track(true);
                d->clearScreen() ;
                d->waitFrames(delay);
                eyet->track(false);
                d->displayText("SAY THE NUMBER LOUD!");
                d->waitForMouseClick();
                d->pushEvent("**************************************") ;
            }
            d->clearScreen();
            d->displayText("Experiment complete. Thank you!");
            d->waitForMouseClick();


          // stop all our ModelComponents
            manager.stop();

        //let's free up the audio files
        for( int i = 0 ; i < 10 ; i++ ){
                Mix_FreeMusic(audioMap[i]);
        }
         //time to close the audio channel
        Mix_CloseAudio();
          // all done!
            return 0;
}

#endif // INVT_HAVE_LIBSDL_IMAGE

