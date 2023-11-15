/*!@file AppPsycho/eyeLinkReplay.C displays the locations of the fixations 
  as the video is playing */
// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
// by the University of Southern California (USC) and the iLab at USC.  //
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
// Primary maintainer for this file: Christian Siagian <siagian@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/eyeLinkReplay.C $
// $Id: $
//

#define RES_W             1024
#define RES_H             768

#define IM_WIDTH        640
#define IM_HEIGHT       480 

#define NUM_CONDITIONS    3
#include <cstdio>
#include "Component/ModelManager.H"
#include "Psycho/EyeLinkAscParser.H"
#include "Media/MPEGStream.H"
#include "GUI/XWinManaged.H"
#include "Raster/Raster.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/DrawOps.H"

// ######################################################################

struct TDexpData
{
  std::string stimuliFileName;
  int blockNum;
  int stimuliIndex;
  Point2D<float> carFollowEndLocation;
  Point2D<float> carFollowStartLocation;
  Point2D<float> freeViewLocation;
};

struct TDexpDataReport
{
  // can add report-wide extra info here

  std::vector<TDexpData> data;
};

rutz::shared_ptr<TDexpDataReport> loadTDexpResult
(std::string fileName, std::string wDir);

std::vector<std::pair<std::string,std::string> > 
getListOfSubjectData(std::string subjListFilename);

//! return the appropriate links to each 
//! stimuli and the condition
//! [stimuli][condition][subject,trialNumber]
std::vector<std::vector<std::vector<std::pair<uint,uint> > > >
fillExperimentData
( std::vector
  <std::pair<rutz::shared_ptr<EyeLinkAscParser>, 
  rutz::shared_ptr<TDexpDataReport> > > subjDataReport,
  std::string fileName);

void createRandDistribution
( std::vector
  <std::pair<rutz::shared_ptr<EyeLinkAscParser>,
  rutz::shared_ptr<TDexpDataReport> > >
  subjDataReport,
  std::vector<std::vector<std::vector<std::pair<uint,uint> > > >
  experimentData,
  std::string subjectFolder);

// ######################################################################
int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager:
  ModelManager manager("EyeLink Replay");

  nub::soft_ref<InputMPEGStream>
    ims(new InputMPEGStream(manager, "Input MPEG Stream", "InputMPEGStream"));
  manager.addSubComponent(ims);

  if (manager.parseCommandLine
      (argc, argv, "[subject list filename] [stimuli list]", 0, 2) == false)
    return(1);

  std::string folder("../TDstimuli/");
  std::string subjectFolder("../TDstimuli/subjects/");
  std::string subjListFilename("../TDstimuli/subjectList.txt");
  std::string expListFilename("../TDstimuli/expList.txt");
  std::string stimuliList("../TDstimuli/listf.txt");
  std::string videoFolder("../TDstimuli/Videos/");
  if(manager.numExtraArgs() >= 2)
    {
      subjListFilename = manager.getExtraArgAs<std::string>(0);
      stimuliList      = manager.getExtraArgAs<std::string>(1);
    }

  std::vector<std::pair<std::string,std::string> > subjDataList = 
    getListOfSubjectData(subjListFilename);

  std::vector
    <std::pair<rutz::shared_ptr<EyeLinkAscParser>,
               rutz::shared_ptr<TDexpDataReport> > >
    subjDataReport(subjDataList.size());

  for(uint i = 0; i < subjDataList.size(); i++)
    {
      std::string ssf = subjDataList[i].first;
      std::string::size_type dpos = ssf.find_first_of('.');
      if(dpos != std::string::npos) ssf = ssf.substr(0,dpos);
      ssf = ssf + std::string("/");

      subjDataReport[i].first.reset
        (new EyeLinkAscParser(subjectFolder+ssf+subjDataList[i].first));
      subjDataReport[i].second = 
        loadTDexpResult(subjectFolder+ssf+subjDataList[i].second, 
                        videoFolder);
    }
  
  // result in terms of experiments
  // list of the eye tracker and response 
  // [stimuli][condition][subject,trialNumber]
  std::vector<std::vector<std::vector<std::pair<uint,uint> > > >
    experimentData = 
    fillExperimentData(subjDataReport, expListFilename);

  LINFO("Finished the experiment related statistics. "
        "Next: [Tatler 2005] distribution");
  Raster::waitForKey();

  // create appropriate spatial biased corrected 
  // random values [Tatler 2005]
  createRandDistribution(subjDataReport, experimentData, subjectFolder);

  LINFO("Below are code to display fixations");
  Raster::waitForKey();
  






  // FIXXX NOTE: Maybe all the subjects !!!!
  uint subjInd = 0;
  rutz::shared_ptr<EyeLinkAscParser> elp = 
    subjDataReport[subjInd].first;
  rutz::shared_ptr<TDexpDataReport> dr =
    subjDataReport[subjInd].second;

  rutz::shared_ptr<XWinManaged> win;

  // for each trials
  uint numTrials = 30;
  for(uint i = 0; i < numTrials; i++)
    {
      // display the videos and result
      ims->setFileName(videoFolder+dr->data[i].stimuliFileName);
      //manager.setOptionValString(&OPT_InputFrameDims,
      //                           convertToString(ims->peekDims()));
      uint width =  ims->peekDims().w();
      uint height = ims->peekDims().h();

      if(win.is_invalid())
        win.reset(new XWinManaged(ims->peekDims(), 0, 0, "Main Window"));
      else win->setDims(ims->peekDims());

      LINFO("\n[[%3d]] stimuli: %s", i, dr->data[i].stimuliFileName.c_str());
      uint bNum = dr->data[i].blockNum;

      // FFFFFFFFFFFFFFFFFIXXXXXXXXXXXXXXXXXXX HACKKKKKKKKKKKKK
      if(bNum != 2) continue;  // want to check car following trials only
      // FFFFFFFFFFFFFFFFFIXXXXXXXXXXXXXXXXXXX HACKKKKKKKKKKKKK

      if(bNum == 1 )
        {
          LINFO("    FV click: %d %d",
                int(float(dr->data[i].freeViewLocation.i)/RES_W * width) , 
                int(float(dr->data[i].freeViewLocation.j)/RES_H * height) );
        }
      if(bNum == 2 )
        {
          LINFO("    CF click: %d %d",
                int(float(dr->data[i].carFollowEndLocation.i)/RES_W * width) , 
                int(float(dr->data[i].carFollowEndLocation.j)/RES_H * height) );
        }

      // open the video for the current trial
      uint flipNum = 0;
      int lastFlipFrame = -1;
      int lastFlipTime  = -1;
      int flipFrame = elp->itsFlipFrame[i][flipNum];
      int flipTime  = elp->itsFlipTime [i][flipNum];
      Image<PixRGB<byte> > ima(ims->peekDims(),ZEROS); 
      uint fixIndex = 0;

      // NOTE THAT WE ONLY REDISPLAY WHEN NEW EVENTS COMES IN
      // Frame flipped, saccades, fixations 
      LINFO("how many gaze points: %" ZU , elp->itsGazeTime[i].size());
      LINFO("flips: %" ZU , elp->itsFlipTime[i].size());

      for(uint j = 0; j < elp->itsGazeTime[i].size(); j++)
        {
          int currTime = elp->itsGazeTime[i][j];

          // check if we need to flip
          if(flipTime <= currTime)
            {
              // update the image 
              for(int k = 0; k < (flipFrame-lastFlipFrame); k++)
                {
                  LDEBUG("[%d][%3d]: %d [%d %d]", 
                         j, flipNum, flipTime, flipFrame, lastFlipFrame);
                  ima = ims->readRGB(); 
                }

              // for display purposes
              Image<PixRGB<byte> > disp = ima;              

              // display all fixations of this flip
              for(uint k = 0; k < elp->itsFlipFixations[i][flipNum].size(); k++)
                {
                  float offsetx = 0.0; 
                  float offsety = 0.0;

                  // note fixations
                  Point2D<int> fixPt
                    (int((elp->itsFlipFixations[i][flipNum][k].i+offsetx)/
                         RES_W * width),
                     int((elp->itsFlipFixations[i][flipNum][k].j+offsety)/
                         RES_H * height)); 

                  LDEBUG("[%3d] [%10.3f %10.3f] -> [%3d %3d]", k,
                         elp->itsFlipFixations[i][flipNum][k].i,
                         elp->itsFlipFixations[i][flipNum][k].j,
                         fixPt.i, fixPt.j);

                  drawDisk(disp, fixPt, 3, PixRGB<byte>(20, 50, 255));
                }
              if(bNum == 2 && flipNum == 0)
                {
                  Point2D<int> st
                    (int(float(dr->data[i].carFollowStartLocation.i)/RES_W * width) , 
                     int(float(dr->data[i].carFollowStartLocation.j)/RES_H * height) );
                  LINFO("%f %f",
                        dr->data[i].carFollowStartLocation.i,
                        dr->data[i].carFollowStartLocation.j);
                  LINFO("start: %d %d", st.i, st.j);
                  drawDisk(disp, st, int(6.0), PixRGB<byte>(255, 0, 0));
                }
              
              if(bNum == 2 && flipNum == elp->itsFlipTime[i].size()-1)
                {
                  Point2D<int> ed
                    (int(float(dr->data[i].carFollowEndLocation.i)/RES_W * width) , 
                     int(float(dr->data[i].carFollowEndLocation.j)/RES_H * height) );
                  LINFO("%f %f",
                        dr->data[i].carFollowEndLocation.i,
                        dr->data[i].carFollowEndLocation.j);
                  LINFO("end:   %d %d", ed.i, ed.j);
                  drawDisk(disp, ed, int(6.0), PixRGB<byte>(0, 255, 0));
 
                }

              if(flipNum%5 == 0 || flipNum >= elp->itsFlipTime[i].size()-1)
                { 
                  win->drawImage(disp,0,0); 
                  //Raster::waitForKey();
                }
              //if(flipNum == 0) Raster::waitForKey();

              //if (ima.initialized() == false) break;  // end of input stream
              lastFlipFrame = flipFrame;
              lastFlipTime  = flipTime;
              
              flipNum++;
              if(flipNum == elp->itsFlipTime[i].size())
                {
                  flipFrame = elp->itsFlipFrame[i][flipNum-1];
                  flipTime  = elp->itsTrialEnd[i];                  
                }
              else
                {
                  flipFrame = elp->itsFlipFrame[i][flipNum];
                  flipTime  = elp->itsFlipTime [i][flipNum];
                }
              LDEBUG("%d %d -------> [%d %d]", 
                     flipNum, flipTime, flipFrame, lastFlipFrame);

              // while these fixations are within the flip range
              bool done = false;
              while(done) //FIXXXXXXXXXXXXXXXXXX should be !done
                {
                  int fixStart = elp->itsFixationStart[i][fixIndex];
                  int fixEnd   = elp->itsFixationEnd  [i][fixIndex];

                  // check for fixations/saccades in this frame
                  //if((lastFlipTime <= fixStart && flipTime >= fixStart) ||
                  //   (lastFlipTime <= fixEnd   && flipTime >= fixEnd  )   )
                  if(lastFlipTime <= fixEnd && flipTime >= fixStart)
                    {
                      // note saccade
                      LINFO("lf: %d fs: %d fe: %d ft: %d", 
                            lastFlipTime, fixStart, fixEnd, flipTime);
                                           
                      // note fixations
                      Point2D<int> fixAvgPt
                        (int(elp->itsFixationAvgLocation[i][fixIndex].i/RES_W * width),
                         int(elp->itsFixationAvgLocation[i][fixIndex].j/RES_H * height)); 

                      float pSize = elp->itsFixationAvgPupilSize[i][fixIndex]; 
                      LINFO("[%3d] pSize: %f [%10.3f %10.3f] -> [%3d %3d]", 
                            fixIndex, pSize,
                            elp->itsFixationAvgLocation[i][fixIndex].i,
                            elp->itsFixationAvgLocation[i][fixIndex].j,
                            fixAvgPt.i, fixAvgPt.j);

                      drawDisk(disp, fixAvgPt, int(6.0), PixRGB<byte>(20, 50, 255));
                      
                      //float x1 = 320.0/1400.0 * width;
                      //float y1 = 370.0/1050.0 * height;
                      //float x2 = 300.0/1400.0 * width;
                      //float y2 = 350.0/1050.0 * height;

                      //LINFO("%f %f || %f %f", 
                      //      x1/640.0*1024, y1/480.0*768, 
                      //      x2/640.0*1024, y2/480.0*768);

                      //drawDisk(disp, Point2D<int>(x1,y1), int(6.0), PixRGB<byte>(255, 0, 0));
                      //drawDisk(disp, Point2D<int>(x2,y2), int(6.0), PixRGB<byte>(0, 255, 0));
                      drawDisk(disp, fixAvgPt, int(6.0), PixRGB<byte>(20, 50, 255));

                      win->drawImage(disp,0,0);
                      Raster::waitForKey();
                    } 

                  // if both FixStart and FixFinish 
                  // is before the lastflipTime
                  // go to next fixation
                  if(lastFlipTime > fixStart && lastFlipTime > fixEnd)
                    {
                      fixIndex++;
                      disp = ima;
                    }
                  else done = true;

                }              
            }
          //LINFO("actually skipping: %d", currTime);
        }

      Raster::waitForKey();                
    }
}

// ######################################################################
rutz::shared_ptr<TDexpDataReport> loadTDexpResult
(std::string fileName, std::string wDir)
{
  rutz::shared_ptr<TDexpDataReport> dataReport(new TDexpDataReport());
  dataReport->data = std::vector<TDexpData>();

  FILE *fp;  char inLine[200]; //char comment[200];

  LINFO("Result file: %s",fileName.c_str());
  if((fp = fopen(fileName.c_str(),"rb")) == NULL)
    { LINFO("not found"); return dataReport; }

  // get the number of segments
  uint nTrials;
  if (fgets(inLine, 200, fp) == NULL) LFATAL("fgets failed"); 
  sscanf(inLine, "%d", &nTrials);
  LDEBUG("Num Trials: %d", nTrials);

  // populate the trial information
  uint cTrial = 0;
  while(fgets(inLine, 200, fp) != NULL)
    {
      TDexpData data;

      // stimuli index and filename
      int index;
      sscanf(inLine, "%d", &index);
      if (fgets(inLine, 200, fp) == NULL) LFATAL("fgets failed");
      std::string sName(inLine);
      sName = sName.substr(0, sName.length()-1);

      // for folder replacement
      // std::string::size_type spos = sName.find_last_of('\\');
      // if(spos != std::string::npos) sName = sName.substr(spos+1);
      // sName = wDir + sName;      

      data.stimuliIndex    = index-1; // matlab code starts at 1
      data.stimuliFileName = sName;
      LDEBUG("[%3d]: %d stimuli file: %s", cTrial, index-1, sName.c_str()); 

      // get the block number
      int bNum;
      if (fgets(inLine, 200, fp) == NULL) LFATAL("fgets failed");
      sscanf(inLine, "%d", &bNum);
      data.blockNum = bNum;
      LDEBUG("[%3d] block number: %d", cTrial, bNum);

      // get the free viewing information
      if(bNum == 1)
        {
          uint gt;
          if (fgets(inLine, 200, fp) == NULL) LFATAL("fgets failed");
          sscanf(inLine, "%d", &gt);

          int x, y;
          if (fgets(inLine, 200, fp) == NULL) LFATAL("fgets failed");
          sscanf(inLine, "%d %d", &x, &y);
          data.freeViewLocation = Point2D<float>(x,y);

          LDEBUG("GT: %d, [%d %d]", gt, x, y);
        }
      else if(bNum == 2)
        {
          float sx, sy;
          if (fgets(inLine, 200, fp) == NULL) LFATAL("fgets failed");
          sscanf(inLine, "%f %f", &sx, &sy);
          data.carFollowStartLocation = Point2D<float>(sx,sy);
          LDEBUG("[%3d] Car follow start location: %f %f", cTrial, sx, sy);

          int x, y;
          if (fgets(inLine, 200, fp) == NULL) LFATAL("fgets failed");
          sscanf(inLine, "%d %d", &x, &y);
          data.carFollowEndLocation = Point2D<float>(x,y);
          LDEBUG("[%3d] Car follow end location: %d %d", cTrial, x,y);              
        }
      else if(bNum == 3)
        {
          if (fgets(inLine, 200, fp) == NULL) LFATAL("fgets failed");
          std::string comment(inLine);
          comment = comment.substr(0, comment.length()-1);
          LDEBUG("[%3d]: %s", cTrial, comment.c_str());
        }

      dataReport->data.push_back(data);
      cTrial++;
    }

  return dataReport;
}

      // // get the information
      // if(bNum == 1)
      //   {
      //     int cNum;
      //     if (fgets(inLine, 200, fp) == NULL) LFATAL("fgets failed");
      //     sscanf(inLine, "%d", &cNum);
      //     //LINFO("[%3d] number of clicks: %d", cTrial, cNum);        

      //     for(int i = 0; i < cNum; i++)
      //       {
      //         int x, y;
      //         if (fgets(inLine, 200, fp) == NULL) LFATAL("fgets failed");
      //         sscanf(inLine, "%d %d", &x, &y);
      //         data.clickLocations.push_back(Point2D<int>(x,y));
      //         //LINFO("[%3d] clicks: %d %d", cTrial, x,y);              
      //       }

      //     data.buildingCount = -1;          
      //   }
      // else
      //   {
      //     int buildingNum;
      //     if (fgets(inLine, 200, fp) == NULL) LFATAL("fgets failed");
      //     sscanf(inLine, "%d", &buildingNum);
      //     data.buildingCount = buildingNum;
      //     //LINFO("[%3d] number of buildings: %d", cTrial, buildingNum);
      //   }

// ######################################################################
std::vector<std::pair<std::string,std::string> > 
getListOfSubjectData(std::string subjListFilename)
{
  std::vector<std::pair<std::string,std::string> > subjList;

  FILE *fp;  char inLine[200]; //char comment[200];

  LINFO("Result file: %s", subjListFilename.c_str());
  if((fp = fopen(subjListFilename.c_str(),"rb")) == NULL)
    { LFATAL("not found"); }

  // populate the subject list
  uint subjListNum = 0;
  while(fgets(inLine, 200, fp) != NULL)
    {
      std::string line(inLine);
      std::string::size_type spos = line.find_first_of(' ');

      if(spos != std::string::npos) 
        {
          uint length = line.length();
          std::string resFname = line.substr(spos+1,length-spos-2);
          std::string ascFname = line.substr(0, spos);
          LINFO("[%3d] asc: %s, res: %s.", subjListNum, 
                ascFname.c_str(), resFname.c_str()); 
          subjList.push_back
            (std::pair<std::string,std::string>
             (ascFname, resFname));
        }
      else LFATAL("Parsing problem for line: \'%s\'", line.c_str());
      subjListNum++;
    }
  return subjList;
}

// ######################################################################
std::vector<std::vector<std::vector<std::pair<uint,uint> > > >
fillExperimentData
( std::vector
  <std::pair<rutz::shared_ptr<EyeLinkAscParser>,
  rutz::shared_ptr<TDexpDataReport> > > subjDataReport,
  std::string fileName)
{
  if(subjDataReport.size() == 0) 
    return std::vector<std::vector<std::vector<std::pair<uint,uint> > > >();

  uint nTrial = subjDataReport[0].second->data.size();
  std::vector<std::vector<std::vector<std::pair<uint,uint> > > >
    expData(nTrial);
  for(uint i = 0; i < nTrial; i++)
    {
      expData[i] = 
        std::vector<std::vector<std::pair<uint,uint> > >(NUM_CONDITIONS);
      for(uint j = 0; j < NUM_CONDITIONS; j++)
        expData[i][j] = std::vector<std::pair<uint,uint> >();
    }

  // go through each subject
  for(uint i = 0; i < subjDataReport.size(); i++)
    {
      // go through each stimuli
      for(uint j = 0; j < subjDataReport[i].second->data.size(); j++)
        {
          uint eNum = subjDataReport[i].second->data[j].stimuliIndex;
          uint cond = subjDataReport[i].second->data[j].blockNum - 1;

          LDEBUG("i: %d j: %d eNum: %d econ: %d", i,j, eNum, cond);
          expData[eNum][cond].push_back(std::pair<uint,uint>(i,j));
        }
    }

  // print
  for(uint i = 0; i < nTrial; i++)
    LINFO("%3d %3d %3d %3d", i+1, 
          int(expData[i][0].size()),
          int(expData[i][1].size()),
          int(expData[i][2].size()));

  Raster::waitForKey();

  // write info to a file    
  FILE *fp = fopen(fileName.c_str(), "wt");
  
  std::string nExp = sformat("%d\n", nTrial);
  fputs(nExp.c_str(), fp);

  std::string nCond = sformat("%d\n", NUM_CONDITIONS);
  fputs(nCond.c_str(), fp);

  for(uint i = 0; i < nTrial; i++)
    {      
      for(uint j = 0; j < NUM_CONDITIONS; j++)
        {
          std::string nSample = 
            sformat("%" ZU "\n", expData[i][j].size());
          fputs(nSample.c_str(), fp);
          for(uint k = 0; k < expData[i][j].size(); k++)
            {
              uint sNum = expData[i][j][k].first;
              uint tNum = expData[i][j][k].second;

              std::string subj = 
                subjDataReport[sNum].first->itsSubjectName;          

              std::string fName = 
                sformat("%s_%d.eyesal-CIOFM\n", 
                        subj.c_str(), tNum+1);

              LDEBUG("%s",fName.c_str());
              fputs (fName.c_str(), fp);
            }
        }
    }
  
  fclose(fp);
  
  return expData;
}

// ######################################################################
void createRandDistribution
( std::vector
  <std::pair<rutz::shared_ptr<EyeLinkAscParser>,
  rutz::shared_ptr<TDexpDataReport> > >
  subjDataReport,
  std::vector<std::vector<std::vector<std::pair<uint,uint> > > >
  experimentData,
  std::string subjectFolder)
{
  uint nSubject = subjDataReport.size();
  uint nTrial   = subjDataReport[0].second->data.size();

  // for each subject
  for(uint i = 0; i < nSubject; i++)
    {
      // get the subject name
      std::string subjName = 
        subjDataReport[i].first->itsSubjectName;          

      // for each trial
      for(uint j = 0; j < nTrial; j++)
        {
          // create a random distribution file
          std::string fileName =
          sformat("%s%s/%s_rdist_%d.txt", 
                  subjectFolder.c_str(), 
                  subjName.c_str(), 
                  subjName.c_str(), j+1);
          LDEBUG("fileName: %s", fileName.c_str());
          FILE *fp = fopen(fileName.c_str(), "wt");
        
          // get the stimuli index and condition
          uint eNum1 = subjDataReport[i].second->data[j].stimuliIndex;
          //uint cond1 = subjDataReport[i].second->data[j].blockNum - 1;

          // add each appropriate saccade
          for(uint ii = 0; ii < nSubject; ii++)
            {
              // skip if same as current subject
              if (i == ii) continue;

              rutz::shared_ptr<EyeLinkAscParser> elp2 = 
                subjDataReport[ii].first;
              rutz::shared_ptr<TDexpDataReport>  dr2  =
                subjDataReport[ii].second;

              // get the subject name
              std::string subjName2 = elp2->itsSubjectName;
              LDEBUG("%s: ", subjName2.c_str());
      
              for(uint jj = 0; jj < nTrial; jj++)
                {
                  // get the stimuli index and condition
                  uint eNum2 = dr2->data[jj].stimuliIndex;
                  //uint cond2 = dr2->data[jj].blockNum - 1;

                  // skip if the stimuli is the same
                  if(eNum1 == eNum2) 
                    { 
                      LDEBUG("skip TRIAL[%3d]: %s", 
                             jj, dr2->data[jj].stimuliFileName.c_str());
                      continue; 
                    }

                  // NOTE: may also want to skip
                  //       if SAME TASK!!!!
                  //if(cond1 == cond2) continue;

                  // get each saccades from the file
                  uint nSaccade = 
                    elp2->itsFixationAvgLocation[jj].size(); 

                  for(uint kk = 0; kk < nSaccade; kk++)
                    {
                      // note fixations
                      float ox = elp2->itsFixationAvgLocation[jj][kk].i;
                      float oy = elp2->itsFixationAvgLocation[jj][kk].j; 

                      int x = int(ox/RES_W * IM_WIDTH);
                      int y = int(oy/RES_H * IM_HEIGHT); 

                      if((x >= 0) && (y <= IM_WIDTH ) &&
                         (y >= 0) && (y <= IM_HEIGHT)   )
                        {
                          std::string coord = sformat("%d %d \n", x, y);
                          fputs (coord.c_str(), fp);
                        }
                      LDEBUG("%s:[%3d][%3d]: (%f %f) ->(%3d %3d)", 
                             subjName2.c_str(), eNum2, kk, ox, oy, x, y);
                    }    
                }
            }
          fclose(fp);
        }
    }
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */
