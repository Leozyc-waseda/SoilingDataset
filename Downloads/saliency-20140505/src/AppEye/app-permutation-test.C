/*!@file AppMedia/app-permutation-test.C Create small images from originals */

//////////////////////////////////////////////////////////////////////////
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the //
// University of Southern California (USC) and the iLab at USC.         //
// See http://iLab.usc.edu for information about this project.          //
//////////////////////////////////////////////////////////////////////////
// Major portions of the iLab Neuromorphic Vision Toolkit are protected //
// under the U.S. patent ``Computation of Intrinsic Perceptual Saliency //
// in Visual Environments, and Applications'' by Christof Koch and      //
// Laurent Itti, California Institute of Technology, 2001 (patent       //
// pending; application number 09/912,225 filed July 23, 2001; see      //
// http://pair.uspto.gov/cgi-bin/final/home.pl for current status).     //
//////////////////////////////////////////////////////////////////////////
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
//////////////////////////////////////////////////////////////////////////
//
// Primary maintainer for this file: David Berg <dberg@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppEye/app-permutation-test.C $

#include "Util/MathFunctions.H"
#include "Psycho/EyesalData.H"
#include "Image/Image.H"
#include <vector>
#include <fstream>
#include <algorithm>
#include <numeric>

#include <ctime>
#include <cstdlib>

#include "Util/StringConversions.H"
#include "Util/StringUtil.H"

#define RANDITER 10000

class iotaGen
{
    public:
    iotaGen (int iv) : current(iv) {}
    int operator ()() {return current++;}
    private:
    int current;
};


//input should be three arguments to eyesal text files  and an output file
int main(int argc, char** argv)
{

    if (argc < 3)
        LFATAL("At least an eyesal file and an output file are needed.");
    else if (argc > 4)
        LFATAL("Right now, I only handle two groups and an output file");
    uint dc = argc-2;
    EyesalData data[dc];
    for (size_t ii = 0;ii < dc;++ii)
        data[ii].setFile(argv[ii+1]);

    srand(time(0));

    //open an output file
    std::ofstream *itsOutFile = new std::ofstream(argv[argc-1]);
    if (itsOutFile->is_open() == false)
        LFATAL("Cannot open '%s' for reading",argv[argc-1]);

    if (dc > 1){//we want to compare two files
            //lets create some giant data vectors
        std::vector<float> model = data[0].getNormSal();
        //position of our original data;
        uint splitpos = model.size();
        std::vector<float> temp = data[1].getNormSal();
        model.insert(model.end(),temp.begin(),temp.end());
        std::vector< std::vector<float> > rand = data[0].getAllNormRandT();
        std::vector< std::vector<float> > tempr =  data[1].getAllNormRandT();
        rand.insert(rand.end(),tempr.begin(),tempr.end());
            //OK now, randomly sample from the data and loop our
            //computations RANDITER times
        for (uint ii = 0; ii < RANDITER; ii++){
            LINFO("Iteration::%d",ii);
                //suffle the data
            std::vector<float> modelr(model.size());
            std::vector< std::vector<float> > randr(rand[0].size());
            std::vector<uint> rshuffle(model.size());
            std::generate(rshuffle.begin(),rshuffle.end(),iotaGen(0));
            if (ii < RANDITER-1) //calculate original groups last
            {
                std::random_shuffle(rshuffle.begin(),rshuffle.end());

                //for testing purposes
                /*std::ifstream *itsinFile = new std::ifstream("order.test");
                if (itsinFile->is_open() == false)
                    PLFATAL("Cannot open '%s'", "adadf");
                std::string line; int linenum = -1;
                while (getline(*itsinFile, line))
                {
                        // one more line that we have read:
                    ++linenum;
                    rshuffle[linenum] = fromStr<int>(line)-1;
                }
                */
                //end test
            }


                //loop through the data and rearange
            for (size_t kk = 0; kk < rand[0].size();++kk)
            {
                randr[kk].resize(model.size());
                for (size_t jj = 0; jj < model.size();++jj)
                {
                    if (kk == 0)
                        modelr[jj] = model[rshuffle[jj]];
                    randr[kk][jj] = rand[rshuffle[jj]][kk];
                }
            }

            //loop through the random sampels
            double meanauc1 = 0;
            double meanauc2 = 0;
            for (size_t jj = 0; jj < randr.size();++jj){
                const float *tm1 = &modelr[0];
                const float *tr1 = &randr[jj][0];
                float auc1 = AUC(tm1,tr1,splitpos,splitpos);

                const float *tm2 = &modelr[splitpos];
                const float *tr2 = &randr[jj][splitpos];
                float auc2 = AUC(tm2,tr2,modelr.size()-splitpos,
                                 modelr.size()-splitpos);
                meanauc1 += auc1;
                meanauc2 += auc2;
            }
            meanauc1 = meanauc1/randr.size();
            meanauc2 = meanauc2/randr.size();
            double outval = (meanauc1-meanauc2);
            //if (outval < 0)
              //outval *= -1;
            (*itsOutFile) << outval << " ";

        }//end RITER
    }//end if dc > 1

//FOR ONLY ONE FILE, TO GET AN ESTIMATE OF CONFIDENCE
    else{
   //lets create some giant data vectors
        std::vector<float> model = data[0].getNormSal();
        //position of our original data;
        uint splitpos = model.size();
        uint halfdata = (uint)floor(splitpos/2);

        std::vector< std::vector<float> > rand = data[0].getAllNormRandT();

            //OK now, randomly sample from the data and loop our
            //computations RANDITER times
        for (uint ii = 0; ii < RANDITER; ii++){
            LINFO("Iteration::%d",ii);
                //suffle the data
            std::vector<float> modelr(model.size());
            std::vector< std::vector<float> > randr(rand[0].size());
            std::vector<uint> rshuffle(model.size());
            std::generate(rshuffle.begin(),rshuffle.end(),iotaGen(0));
            uint ldata = model.size();
            if (ii < RANDITER-1)
            {//calculate original groups last
                std::random_shuffle(rshuffle.begin(),rshuffle.end());
                ldata = halfdata;
            }

                //loop through the data and rearange
            for (size_t kk = 0; kk < rand[0].size();++kk)
            {
                randr[kk].resize(ldata);
                for (size_t jj = 0; jj < ldata;++jj)
                {
                    if (kk == 0)
                        modelr[jj] = model[rshuffle[jj]];
                    randr[kk][jj] = rand[rshuffle[jj]][kk];
                }
            }

            //loop through the random sampels
            double meanauc = 0;


            for (size_t jj = 0; jj < randr.size();++jj){
                const float *tm1 = &modelr[0];
                const float *tr1 = &randr[jj][0];
                float auc = AUC(tm1,tr1,ldata,ldata);
                meanauc += auc;
            }
            meanauc = meanauc/randr.size();
            (*itsOutFile) << meanauc << " ";

        }
    }



itsOutFile->close();
}//END MAIN

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
