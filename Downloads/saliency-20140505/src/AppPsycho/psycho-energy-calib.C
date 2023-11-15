/*!@file AppPsycho/psycho-math-op.C Psychophysics test to measure the influence of eyemovement on memory task performance */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-energy-calib.C $


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

ModelManager manager("Psycho-Calib");
nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
map<uint,uint> testMap ;
map<string,string> argMap ;

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

        com += "\nlogfile=[logfilename.psy] {default = psycho-stroop-concurrent.psy}\n" ;
        com += "\nmemo=[a_string_without_white_space]\n";
        com += "\nx-grid=[>0](half minus one of num of points along x direction){default=5} \n";
        com += "\nsaccade=[>0](half minus one of number of saccades to be made at each point){default=5} \n";
        com += "\nsubject=[subject_name] \n" ;
        return com ;
}


extern "C" int main(const int argc, char** argv)
{

        MYLOGVERB = LOG_INFO;  // suppress debug messages
        //let's push the initial value for the parameters
        argMap["experiment"]="testing-calibration-for-energy-distribution";
        argMap["logfile"]="psycho-energy-calib.psy" ;
        argMap["subject"]="" ;
        argMap["memo"]="" ;
        argMap["x-grid"] = "5";
        argMap["saccade"] = "5";

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
        int x_g = atoi(argMap["x-grid"].c_str()) ;
        int y_g = atoi(argMap["saccade"].c_str()) ;
//          SDL_Rect offset ;
  // let's get all our ModelComponent instances started:
            manager.start();
            for(map<string,string>::iterator it= argMap.begin(); it!= argMap.end() ; ++it) d->pushEvent("arg:"+ it->first+" value:"+it->second ) ;
  // let's display an ISCAN calibration grid:
            d->clearScreen();
            d->displayISCANcalib();
            d->waitForMouseClick();
            d->displayText("CLICK LEFT button to calibrate; RIGHT to skip");
            int cl = d->waitForMouseClick();
            if (cl == 1) d->displayEyeTrackerCalibration(3,5,1 , true);
            d->clearScreen();
            d->displayText("Here the experiment starts! click to start!");
            d->waitForMouseClick() ;
            d->clearScreen();
            //SDL_Surface* mark= getABlankSurface(5,5);
            int x0 = d->getWidth();
            for(int i = - x_g ; i < x_g + 1 ; i++){
                for(int j = -y_g ; j < y_g +1 ; j++){
                        int y = d->getHeight()/2;
                        int x1 = x0+ i*32;
                        int x2 = x1 + j*32;
                        d->displayFixation();
                        d->waitForMouseClick() ;
                        std::string imst = "===== Showing image: im"+stringify(x_g)+"-"+stringify(y_g)+".png =====";
                        d->pushEvent(imst);
                        d->displayFixationBlink(x1,y,3,2);
                        eyet->track(true);
                        d->displayFixationBlink(x1,y,16,2);
                        d->displayFixationBlink(x2,y,16,2);
                        eyet->track(false);
                        d->clearScreen();

                }
            }

            d->waitForMouseClick();
            d->clearScreen();

            d->clearScreen();
            d->displayText("Experiment complete. Thank you!");
            d->waitForMouseClick();


          // stop all our ModelComponents
            manager.stop();


            return 0;
}

#endif // INVT_HAVE_LIBSDL_IMAGE

