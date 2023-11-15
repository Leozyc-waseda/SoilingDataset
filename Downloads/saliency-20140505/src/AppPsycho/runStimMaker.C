/*!@file AppPsycho/runStimMaker.C make different kind of visual test stimuli
 */

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
// Primary maintainer for this file: T. Nathan Mundhenk <mundhenk@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/runStimMaker.C $
// $Id: runStimMaker.C 14376 2011-01-11 02:44:34Z pez $
//

#ifndef RUN_STIM_MAKER_C_DEFINED
#define RUN_STIM_MAKER_C_DEFINED

#include "Psycho/StimMaker.H"
#define STIM_DEMO_MODE false
#define STIM_MAKE_SURPRISE_SET true

using std::string;

//**********************************************************************

int main(const int argc, const char **argv)
{
  if(STIM_DEMO_MODE)
  {
    LINFO("RUNNING IN DEMO MODE");
    StimMakerParam stim;
    StimMaker      maker(512,512,120,SM_COLOR_BLACK);

    LINFO("RUNNING FIRST DEMO");
    stim.setDemoParams1();
    maker.SM_makeUniformStim(stim);
    std::vector<Image<PixRGB<float> > > stim1 = maker.SM_getStim();
    std::vector<Image<PixRGB<float> > > gt1   = maker.SM_getGroundTruth();

    LINFO("RUNNING SECOND DEMO");
    maker.SM_init(SM_COLOR_BLACK);
    stim.setDemoParams2();
    maker.SM_makeUniformStim(stim);
    std::vector<Image<PixRGB<float> > > stim2 = maker.SM_getStim();
    std::vector<Image<PixRGB<float> > > gt2   = maker.SM_getGroundTruth();

    LINFO("RUNNING THIRD DEMO");
    maker.SM_init(SM_COLOR_BLACK);
    stim.setDemoParams3();
    maker.SM_makeUniformStim(stim);
    std::vector<Image<PixRGB<float> > > stim3 = maker.SM_getStim();
    std::vector<Image<PixRGB<float> > > gt3   = maker.SM_getGroundTruth();

    LINFO("RUNNING FOURTH DEMO");
    maker.SM_init(SM_COLOR_BLACK);
    stim.setDemoParams4();
    maker.SM_makeUniformStim(stim);
    std::vector<Image<PixRGB<float> > > stim4 = maker.SM_getStim();
    std::vector<Image<PixRGB<float> > > gt4   = maker.SM_getGroundTruth();

    LINFO("WRITING OUT FILES");
    for(uint i = 0; i < stim1.size(); i++)
    {
      const uint frameNumber = i;

      char c[100];
      string a, Myname;
      string b = ".";
      if(frameNumber < 10)
        sprintf(c,"00000%d",frameNumber);
      else if(frameNumber < 100)
        sprintf(c,"0000%d",frameNumber);
      else if(frameNumber < 1000)
        sprintf(c,"000%d",frameNumber);
      else if(frameNumber < 10000)
        sprintf(c,"00%d",frameNumber);
      else if(frameNumber < 100000)
        sprintf(c,"0%d",frameNumber);
      else
        sprintf(c,"%d",frameNumber);

      a      = "frame.set1";
      Myname = a + b + c;

      Raster::WriteRGB(stim1[i],Myname,RASFMT_PNM);


      a      = "frame.set2";
      Myname = a + b + c;
      Raster::WriteRGB(stim2[i],Myname,RASFMT_PNM);
      a      = "frame.set3";
      Myname = a + b + c;
      Raster::WriteRGB(stim3[i],Myname,RASFMT_PNM);
      a      = "frame.set4";
      Myname = a + b + c;
      Raster::WriteRGB(stim4[i],Myname,RASFMT_PNM);


      a      = "frame.GT1";
      Myname = a + b + c;

      Raster::WriteRGB(gt1[i],Myname,RASFMT_PNM);


      a      = "frame.GT2";
      Myname = a + b + c;
      Raster::WriteRGB(gt2[i],Myname,RASFMT_PNM);
      a      = "frame.GT3";
      Myname = a + b + c;

      Raster::WriteRGB(gt3[i],Myname,RASFMT_PNM);
      a      = "frame.GT4";
      Myname = a + b + c;
      Raster::WriteRGB(gt4[i],Myname,RASFMT_PNM);


    }
  }
  if(STIM_MAKE_SURPRISE_SET)
  {
    LINFO("RUNNING SURPRISE SET");
    StimMakerParam stim;
    StimMaker      maker(256,256,120,SM_COLOR_BLACK);

    stim.setBasicParams1();

    std::vector<unsigned char> distColor(2,0);
    distColor[0]    = SM_COLOR_BLUE;
    distColor[1]    = SM_COLOR_YELLOW;
    std::vector<unsigned char> shape(2,0);
    shape[0]        = SM_STIM_CROSS;
    shape[1]        = SM_STIM_DISK;
    std::vector<float>         distOri(2,0.0F);
    distOri[0]      = M_PI/4.0F;
    distOri[1]      = 0.0F;
    std::vector<float>         jitter(2,0.0F);
    jitter[0]       = 0.0F;
    jitter[1]       = 0.2F;

    stim.SMP_targetColor                  = SM_COLOR_BLUE;
    stim.SMP_targetOri                    = 0.0F;
    stim.SMP_shapeOrientationJitter       = 0.0F;
    stim.SMP_shapePositionJitterStatic    = 0.2F;
    stim.SMP_shapeOrientationJitterStatic = 0.2F;


    for(unsigned char p = 0; p < 2; p++)
    {
      LINFO(">SHAPE %d - %d",p,shape[p]);
      stim.SMP_distShape   = shape[p];
      stim.SMP_targetShape = shape[p];
      for(unsigned char i = 0; i < 3; i++)
      {
        LINFO(">> DIST RATE %d",i);
        stim.SMP_distRate = i;
        for(unsigned char j = 0; j < 3; j++)
        {
          LINFO(">>> TARGET RATE %d",j);
          stim.SMP_targetRate = j;
          for(unsigned char k = 0; k < 3; k++)
          {
            LINFO(">>>> DIST STATE %d",k);
            stim.SMP_distState = k;
            for(unsigned char l = 0; l < 3; l++)
            {
              LINFO(">>>>> TARGET STATE %d",l);
              stim.SMP_targetState = l;
              for(unsigned char n = 0; n < 2; n++)
              {
                LINFO(">>>>>> JITTER %d - %f",n,jitter[n]);
                stim.SMP_shapePositionJitter = jitter[n];
                for(unsigned char o = 0; o < 2; o++)
                {
                  LINFO(">>>>>>> DIST COLOR %d - %d",o,distColor[o]);
                  stim.SMP_distColor = distColor[o];
                  float ori    = 0.0F;
                  for(unsigned char m = 0; m < 2; m++)
                  {

                    if(stim.SMP_distShape == SM_STIM_CROSS)
                    {
                      ori    = distOri[m];
                    }
                    else
                    {
                      ori    = 0.0F;
                    }
                    LINFO(">>>>>>>> DIST ORI %d - %f",m,ori);
                    if((m < 1) || (stim.SMP_distShape == SM_STIM_CROSS))
                    {
                      LINFO("STIM OK");
                      stim.SMP_distOri = ori;
                      maker.SM_makeUniformStim(stim);
                      std::vector<Image<PixRGB<float> > > stim
                        = maker.SM_getStim();
                      std::vector<Image<PixRGB<float> > > gt
                        = maker.SM_getGroundTruth();

                      for(uint x = 0; x < stim.size(); x++)
                      {
                        const uint frameNumber = x;

                        char c[100];
                        char d[100];
                        string a, Myname;
                        string b = ".";
                        if(frameNumber < 10)
                          sprintf(c,"00000%d",frameNumber);
                        else if(frameNumber < 100)
                          sprintf(c,"0000%d",frameNumber);
                        else if(frameNumber < 1000)
                          sprintf(c,"000%d",frameNumber);
                        else if(frameNumber < 10000)
                          sprintf(c,"00%d",frameNumber);
                        else if(frameNumber < 100000)
                          sprintf(c,"0%d",frameNumber);
                        else
                          sprintf(c,"%d",frameNumber);

                        sprintf(d,"%d.%d.%d.%d.%d.%d.%d.%d",p,i,j,k,l,n,o,m);

                        a      = "surprise.set";
                        Myname = a + b + d + b + c;
                        Raster::WriteRGB(stim[x],Myname,RASFMT_PNG);
                        a      = "surprise.GT4";
                        Myname = a + b + d + b + c;
                        Raster::WriteRGB(gt[x],Myname,RASFMT_PNG);
                      }
                      maker.SM_init(SM_COLOR_BLACK);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  LINFO("DONE");
}

#endif
