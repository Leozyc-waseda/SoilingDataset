/*!@file GameBoard/psycho-direction.C overlays the markup on top of the image */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/GameBoard/psycho-direction.C $
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
#include "GameBoard/basic-graphics.H"
#include "GameBoard/resize.h"
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <map>
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

#define PI 3.14159265

using namespace std;
map<string,string> argMap ;


//pushes back the name of files in the directory into the given vector
int getdir (string dir, vector<string> &files ,string ext)
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
                        if(extension.compare(ext)== 0 )
                                files.push_back(dir+"/"+fn);
                }
        }
        closedir(dp);
        return 0;
}

string getCorrespondingDataFile(string fn, vector<string> files){
        uint pos = fn.find_last_of("/");
        string lp = fn.substr(pos+1) ;
        lp = lp.substr(0,lp.find("."));
        for( uint i = 0 ; i < files.size() ; i++ ){
                uint posd = files[i].find_last_of("/");
                string lpd = files[i].substr(posd+1) ;
                lpd = lpd.substr(0,lpd.find("."));
                if(lpd.compare(lp) == 0) return files[i] ;
        }
        return "" ;
}
void overlay_rect(SDL_Surface* surf,Point2D<int> location , Point2D<int> size , Uint32 bc){
        Point2D<int> corner = Point2D<int>(location.i- size.i/2 , location.j - size.j/2) ;
        drawRectangle(surf,bc,corner.i,corner.j,size.i -1,size.j -1 ,1);

}


int addArgument(const string st,const string delim="="){
        int i = st.find(delim) ;
        argMap[st.substr(0,i)] = st.substr(i+1);

        return 0 ;
}

std::string getArgumentValue(string arg){
        return argMap[arg] ;
}

std::string getUsageComment(){

        string com = string("\nlist of arguments : \n");

        com += "\ndelay=[>=0] (the delaly for overlaying marks) {default=0}\n";
        com += "\nlogfile=[logfilename.psy] {default = eyemarkup.psy}\n" ;
        com += "\nmode=[1..2] (1 for static , 2 for dynamic) {default=2}\n" ;
        com += "\nimage-dir=[path to directory of images]\n" ;
        com += "\ndata-dir=[path to directory of .ceyeS files]\n" ;
        return com ;
}

void findDirection(Point2D<float> p , Point2D<float> c , map<double,double>& dirMap){
        bool flag = true ;
        if(c.i == p.i) flag=false;
        double myTan = (double)(-c.j + p.j)/(double)(c.i-p.i);
        double dis = sqrt((-c.j + p.j)*(-c.j + p.j) + (c.i-p.i)*(c.i-p.i));
        double myCos = (double)(c.i-p.i)/dis;
        double direction = 0.0 ;

        if(flag){
                if( myCos<=0 ){
                        if(myTan<=0) {direction = atan(myTan) + PI ;}else{ direction = atan(myTan) - PI ;}
                } else{
                        direction = atan(myTan) ;
                }
        }else{
                if(c.j>p.j){
                        direction = -PI/2 ;
                } else{
                        direction = PI/2 ;
                }
        }

        map<double,double>::iterator it = dirMap.find(direction);
        if( it!= dirMap.end()) {
               // cout<<"here"<<endl ;
                dirMap[direction] = dirMap[direction]+dis ;
        }else{
                dirMap[direction] = dis ;
               // cout<<"there  " << direction << "   "<< dis <<endl ;
        }

}

extern "C" int main(const int argc, char** argv)
{
        ModelManager manager("Psycho Skin");
        nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));

        MYLOGVERB = LOG_INFO;  // suppress debug messages
        argMap["mode"] = "2" ;
        argMap["logfile"]="eyemarkup.psy" ;
        argMap["image-dir"]=".";
        argMap["data-dir"] = "." ;
        argMap["delay"]="0" ;

        manager.addSubComponent(d);
        nub::soft_ref<EventLog> el(new EventLog(manager));
        manager.addSubComponent(el);
        d->setEventLog(el);
        if (manager.parseCommandLine(argc, argv,
            "at least one argument needed", 1, -1)==false){
                    cout<<getUsageComment()<<endl;
                    return(1);
        }

        for(uint i = 0 ; i < manager.numExtraArgs() ; i++){
                    addArgument(manager.getExtraArg(i),std::string("=")) ;
        }
        // Parse command-line:
        manager.setOptionValString(&OPT_EventLogFileName, argMap["logfile"]);
        manager.start();

        string dir = argMap["image-dir"];
        vector<string> imfiles = vector<string>();
        getdir(dir,imfiles,".png");

        dir = argMap["data-dir"];
        vector<string> datafiles = vector<string>();
        getdir(dir,datafiles,".eyes");

        while(imfiles.size() > 0){
                string imfilename = imfiles[0] ;
                imfiles.erase(imfiles.begin());
                LINFO(imfilename.c_str());

        //let's define a bunch of colors for different events and put them in a map
//         fixation:                   0 blue
//         saccade:                    1 green
//         blink/Artifact:             2 red
//         Saccade during Blink:       3 green .-
//         smooth pursuit:             4 magenta
//         drift/misclassification:    5 black
                Uint32 blue = d->getUint32color(PixRGB<byte>(0, 0, 255));
                Uint32 green = d->getUint32color(PixRGB<byte>(0,255,0) );
                Uint32 red = d->getUint32color(PixRGB<byte>(255,0,0) );
                Uint32 saccade_green = d->getUint32color(PixRGB<byte>(0,150,0) );
                Uint32 magenta = d->getUint32color(PixRGB<byte>(0,255,255) );
                Uint32 black = d->getUint32color(PixRGB<byte>(0,0,0) );
                map<string,Uint32> colormap = map<string,Uint32>() ;
                colormap["0"] = blue ; colormap["1"] = green ;colormap["2"] = red ;
                colormap["3"] = saccade_green ; colormap["4"]=magenta ; colormap["5"]= black ;

        //now let's take in the corresponding .ceyeS file
                string datafilename = getCorrespondingDataFile(imfilename,datafiles);
                if(datafilename.size()!=0){

                        SDL_Surface* im = load_image(imfilename) ;
                        float rsf = min((float)d->getWidth()/(float)im->w ,  (float)d->getHeight()/(float)im->h ) ;
                        im = SDL_Resize(im , rsf, 5) ;
                        SDL_Rect offset ;
                        offset.x= (d->getWidth() - im->w) /2;
                        offset.y=(d-> getHeight() - im->h) /2;
                      //  int maxX = (d->getWidth()-10);//*rsf;
                       // int maxY = (d->getHeight()-10);//*rsf ;
                        ifstream inFile( datafilename.c_str() , ios::in);
                        if (! inFile)
                        {
                                LINFO("profile '%s' not found!" ,   datafilename.c_str());
                                return -1;
                        }
                        Point2D<int> loc ;
                        Point2D<int> rectsize = Point2D<int>(6,6) ;
                        Point2D<float> preLoc ;
                        preLoc.i = -1.0 ;
                        Point2D<float> curLoc ;
                        char ch[1000];
                        map<double,double> dirDistMap;
                        while (inFile.getline(ch , 1000) ){
                                string line = ch;
                                //cout<<line<<endl;
                                if(line.compare("NaN NaN NaN 0.0")!=0){
                                uint pos = line.find(" ");
                                curLoc.i = atof(line.substr(0,pos).c_str()) ;
                                loc.i = (int) (rsf*curLoc.i) ;
                                line = line.substr(pos+1);
                                pos = line.find(" ");
                                curLoc.j = atof(line.substr(0,pos).c_str()) ;
                                loc.j = (int) (rsf*curLoc.j) ;

                               // string colorcode = line.substr(line.size()-1) ;
                                //if(colorcode.compare("2")!=0){
                                        if(preLoc.i!=-1.0){
                                                findDirection(preLoc,curLoc,dirDistMap);
                                        }
                                        preLoc.i = curLoc.i ;
                                        preLoc.j = curLoc.j ;
                                //}else{
                                  //      preLoc.i = -1.0 ;
                                //}
                                // if (loc.i > 4 && loc.j > 4 && loc.i < maxX && loc.j < maxY)
                                       //  overlay_rect(im,loc,rectsize,colormap["1"]);

                                 if(argMap["mode"].compare("2")==0)
                                         d->waitFrames(atoi(argMap["delay"].c_str()));
                                 d->displaySDLSurfacePatch(im , &offset,NULL , -2,false, true) ;
                                }else{preLoc.i = -1.0 ;}
                        }

                        map<Point2D<double>,double> binDirDistMap ;
                        map<double, double>::iterator iter;

//                       cout<<"size " <<dirDistMap.size()<<endl ;
//                         for (iter=dirDistMap.begin(); iter != dirDistMap.end(); ++iter) {
//                                         cout<<"direction "<<iter->first<<"  distance "<<iter->second<<endl ;
//                                        // if (iter->first >= a && iter->second < a+PI/32.0) binDirDistMap[a] = binDirDistMap[a]+iter->second;
//                         }

                        double tD= 0.0 ;

                        int numOfSec = 64 ;
                        for(int i = 0  ; i < numOfSec ; i++){
                                double a = (i- numOfSec/2 )*PI/(numOfSec/2) ;
                                binDirDistMap[Point2D<double>(a,a+ PI/(numOfSec/2))] = 0.0 ;
                        }
                        map<Point2D<double>,double>::iterator fiter ;
                        for(fiter=binDirDistMap.begin() ; fiter != binDirDistMap.end() ; ++fiter){
                                Point2D<double> p = fiter->first ;
                                double v = 0.0 ;
                                for(iter = dirDistMap.begin() ; iter != dirDistMap.end() ; ++iter){
                                        double ov = iter->first ;
                                        double od = iter->second ;
                                        if(ov>p.i && ov<=p.j) v+= od ;
                                }
                                binDirDistMap[p] = v ;
                                tD+=v ;
                                cout<<p.j<<" "<<v<<endl ;
                        }
//                         for(int i = 0  ; i < 64 ; i++){
//                                 double a = (i-32)*PI/32 ;
//                                 for (iter=dirDistMap.begin(); iter != dirDistMap.end(); ++iter) {
//                                         if (iter->first > a && iter->second <= a+PI/32.0) binDirDistMap[a] += iter->second;
//                                         tD += iter->second ;
//                                 }
//                         }
//                         for (iter=binDirDistMap.begin(); iter != binDirDistMap.end(); ++iter) {
//                                        cout<<iter->first<<"         "<<iter->second<<endl ;
//                         }
                       // cout<<"total :"<<tD <<endl ;
                        d->waitForMouseClick() ;
                        dumpSurface(im) ;

                }




        }
        manager.stop();
        return 0 ;

}

