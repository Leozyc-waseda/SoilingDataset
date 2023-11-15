/*!@file Beobot/GSNavResult.H read and summarize result files           */
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
// Primary maintainer for this file: Christian Siagian <siagian@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/GSnavResult.H $
// $Id: GSnavResult.H 15310 2012-06-01 02:29:24Z itti $

// ######################################################################

#ifndef BEOBOT_GSNAV_RESULT_H_DEFINED
#define BEOBOT_GSNAV_RESULT_H_DEFINED

#include "Image/Image.H"
#include "Image/MathOps.H"      // for inPlaceNormalize()

// ######################################################################

/*! This class is to read the GSnav result file                        */
class GSnavResult
{
public:

  // input information (on segment)
  std::vector<double> segError;
  std::vector<uint>   segCount;
  std::vector<uint>   eSegCount;

  std::vector<uint> nSegObject;
  std::vector<uint> nSegObjectSearch;

  std::vector<uint> nSegObjectFound;
  std::vector<uint> nSegObjectFoundSearch;

  std::vector<uint> nSegObjectNotFound;
  std::vector<uint> nSegObjectNotFoundSearch;

  // input information (on frame)
  std::vector<double> error;
  double stdevError;

  std::vector<uint> nObject;
  double stdevNObject;
  std::vector<uint> nObjectSearch;

  std::vector<uint> nObjectFound;
  double stdevNObjectFound;
  std::vector<uint> nObjectFoundSearch;

  std::vector<uint> nObjectNotFound;
  double stdevNObjectNotFound;
  std::vector<uint> nObjectNotFoundSearch;

  // file information
  uint nsegment;
  std::string savePrefix;

  // total across segment information
  double terror;  uint tcount;
  uint tfobj;     uint tfsearch;
  uint tnfobj;    uint tnfsearch;
  uint tobj;      uint tsearch;
  uint itsSearchCount;

  inline void combine(std::string inSavePrefix);
  inline void read(std::string inSavePrefix, uint inNsegment);
  inline void createSummaryResult();
  inline void getSearchResult(std::vector<uint> index);
};

// ######################################################################
inline void GSnavResult::combine(std::string inSavePrefix)
{
  savePrefix = inSavePrefix;

  // open the result file
  std::string resFName = savePrefix + sformat("_results.txt");
  std::string resGSfName = savePrefix + sformat("_GS_results.txt");
  FILE *rfp; LINFO("res files: %s", resFName.c_str());
  if((rfp = fopen(resFName.c_str(),"rb")) == NULL)LFATAL("not found");
  FILE *gfp; LINFO("GS res files: %s", resGSfName.c_str());
  if((gfp = fopen(resGSfName.c_str(),"rb")) == NULL)LFATAL("not found");

  // get each line from result file
  std::vector<std::string> rfpLines;
  char inLine[200];
  while(fgets(inLine, 200, rfp) != NULL)
    {
      rfpLines.push_back(std::string(inLine));
    }

  // get each line from GS result file
  uint lind = 0;
  std::vector<std::string> cfpLines;

  std::string resCfName = savePrefix + sformat("_comb_results.txt");
  FILE *rFile = fopen(resCfName.c_str(), "at");
  if (rFile == NULL) LFATAL("can't create res file: %s",
                            resCfName.c_str());

  while(fgets(inLine, 200, gfp) != NULL)
    {
      // read the line
      char tLine[200]; uint cind;
      sscanf(inLine, "%d %s", &cind, tLine);
      LDEBUG("%6d <<%s>>", cind, tLine);

      // copy the non end-search indexes
      for(uint i = lind; i < cind; i++)
        {
          cfpLines.push_back(rfpLines[i]);
          LDEBUG("%s", rfpLines[i].c_str());
          fputs(rfpLines[i].c_str(), rFile);
        }

      // append the two lines of the end-search index
      std::string temp(inLine);
      std::string::size_type lzpos = rfpLines[cind].find_last_of('0');
      std::string::size_type pos = temp.find_first_not_of(' ');
      temp = temp.substr(pos);
      pos = temp.find_first_of(' ');
      temp = rfpLines[cind].substr(0, lzpos) + temp.substr(pos+1);
      cfpLines.push_back(temp);
      LDEBUG("[%d] %s", itsSearchCount, temp.c_str());

      fputs(temp.c_str(), rFile);

      lind = cind+1;
    }
  fclose (rFile);
}

// ######################################################################
inline void GSnavResult::read(std::string inSavePrefix, uint inNsegment)
{
  savePrefix = inSavePrefix;
  nsegment = inNsegment;

  // open the result file
  //std::string resFName = savePrefix + sformat("_comb_results.txt");
  std::string resFName = savePrefix + sformat("_results.txt");
  FILE *fp; LINFO("result file: %s", resFName.c_str());
  if((fp = fopen(resFName.c_str(),"rb")) == NULL)LFATAL("not found");

  // initialize information storage
  segError.resize(nsegment);
  segCount.resize(nsegment);
  eSegCount.resize(nsegment);

  nSegObject.resize(nsegment);
  nSegObjectSearch.resize(nsegment);

  nSegObjectFound.resize(nsegment);
  nSegObjectFoundSearch.resize(nsegment);

  nSegObjectNotFound.resize(nsegment);
  nSegObjectNotFoundSearch.resize(nsegment);

  for(uint i = 0;  i < nsegment; i++) segError[i] = 0.0F;
  for(uint i = 0;  i < nsegment; i++) segCount[i] = 0;
  for(uint i = 0;  i < nsegment; i++) eSegCount[i] = 0;

  for(uint i = 0;  i < nsegment; i++) nSegObjectFound[i] = 0;
  for(uint i = 0;  i < nsegment; i++) nSegObjectFoundSearch[i] = 0;

  for(uint i = 0;  i < nsegment; i++) nSegObjectNotFound[i] = 0;
  for(uint i = 0;  i < nsegment; i++) nSegObjectNotFoundSearch[i] = 0;

  // get each line result
  char inLine[200];
  itsSearchCount = 0;
  while(fgets(inLine, 200, fp) != NULL)
    {
      int index; uint ninput;
      uint snumGT; float ltravGT;
      uint snumRes; float ltravRes; float err;

      // read the line
      char tLine[200];
      sscanf(inLine, "%d %d %f %d %f %f %d %s", &index,
             &snumGT, &ltravGT, &snumRes, &ltravRes, &err, &ninput, tLine);
      LDEBUG("%6d %6d %10.6f %6d %10.6f %10.6f %6d", index,
             snumGT, ltravGT, snumRes, ltravRes, err, ninput);

      if(ninput != 0) itsSearchCount++;

      segError[snumGT] += double(err);
      segCount[snumGT]++;
      error.push_back(err);

      if(snumGT != snumRes) eSegCount[snumGT]++;

      // get the individual object search
      uint nFound  = 0; uint nFoundSearch    = 0;
      uint nNFound = 0; uint nNotFoundSearch = 0;
      std::string line(inLine);
      for(uint i = 0; i < ninput; i++)
        {
          // do it backwards
          int pos = line.find_last_of(' ');
          uint tnsearch = atoi(line.substr(pos+1).c_str());
          line = line.substr(0, pos);
          line = line.substr(0, line.find_last_not_of(' ')+1);

          pos = line.find_last_of(' ');
          bool tfound = !(line.substr(pos+1).compare("1"));
          line = line.substr(0, pos);
          line = line.substr(0, line.find_last_not_of(' ')+1);

          LDEBUG("%d %d", tfound, tnsearch);

          if(tfound)
            {
              nSegObjectFound[snumGT]++;
              nSegObjectFoundSearch[snumGT] += tnsearch;
              nFound++;
              nFoundSearch += tnsearch;
            }
          else
            {
              nSegObjectNotFound[snumGT]++;
              nSegObjectNotFoundSearch[snumGT] += tnsearch;
              nNFound++;
              nNotFoundSearch += tnsearch;
            }

          LDEBUG("%d %d - %d %d",
                nSegObjectFound[snumGT], nSegObjectFoundSearch[snumGT],
                nSegObjectNotFound[snumGT], nSegObjectNotFoundSearch[snumGT]);
        }

      // store for data of each frame
      nObjectFound.push_back(nFound);
      nObjectNotFound.push_back(nNFound);
      nObject.push_back(nFound + nNFound);

      nObjectFoundSearch.push_back(nFoundSearch);
      nObjectNotFoundSearch.push_back(nNotFoundSearch);
      nObjectSearch.push_back(nFoundSearch + nNotFoundSearch);
    }
  fclose(fp);

  // get across segment total
  terror = 0.0; tcount    = 0;
  tfobj  = 0;   tfsearch  = 0;
  tnfobj = 0;   tnfsearch = 0;
  tobj   = 0;   tsearch   = 0;
  for(uint i = 0; i < nsegment; i++)
    {
      terror+= segError[i];
      tcount+= segCount[i];

      // search related numbers
      nSegObject[i] = nSegObjectFound[i] + nSegObjectNotFound[i];
      nSegObjectSearch[i] =
        nSegObjectFoundSearch[i] + nSegObjectNotFoundSearch[i];

      tfobj     += nSegObjectFound[i];
      tfsearch  += nSegObjectFoundSearch[i];
      tnfobj    += nSegObjectNotFound[i];
      tnfsearch += nSegObjectNotFoundSearch[i];
      tobj      += nSegObject[i];
      tsearch   += nSegObjectSearch[i];
    }

  // get total standard deviation
  Image<double> t1(1,nObjectFound.size(),NO_INIT);
  for(uint i = 0; i < nObjectFound.size(); i++)
    t1.setVal(0, i, double(nObjectFound[i]));
  stdevNObjectFound = stdev(t1);

  for(uint i = 0; i < nObjectNotFound.size(); i++)
    t1.setVal(0, i, double(nObjectNotFound[i]));
  stdevNObjectNotFound = stdev(t1);

  for(uint i = 0; i < nObjectFound.size(); i++)
    t1.setVal(0, i, double(nObject[i]));
  stdevNObject = stdev(t1);

  Image<double> t2(1,error.size(),NO_INIT);
  for(uint i = 0; i < error.size(); i++) t2.setVal(0, i, error[i]);
  stdevError = stdev(t2);
}

// ######################################################################
inline void GSnavResult::createSummaryResult()
{
  std::string sresFName = savePrefix + sformat("_summary_results.txt");
  FILE *rfp; LINFO("result file: %s", sresFName.c_str());
  if((rfp = fopen(sresFName.c_str(),"wt")) == NULL)LFATAL("not found");

  // segment line
  for(uint i = 0; i < nsegment; i++)
    {
      std::string line(sformat("[%6d]:     0        0.0      0",i));
      if(segCount[i] > 0)
        {
          line = sformat
            ("[%6d]: %6d %10.6f %6d  -- %6d %10d: %10.3f + %6d %10d: %10.3f"
             " = %6d %10d %10.3f",
             i, segCount[i], segError[i]/segCount[i], eSegCount[i],
             nSegObjectFound[i],  nSegObjectFoundSearch[i],
             nSegObjectFoundSearch[i]/double(nSegObjectFound[i]),
             nSegObjectNotFound[i], nSegObjectNotFoundSearch[i],
             nSegObjectNotFoundSearch[i]/double(nSegObjectNotFound[i]),
             nSegObject[i], nSegObjectSearch[i],
             double(nSegObjectSearch[i])/nSegObject[i]);
        }
      LINFO("%s",line.c_str());
      line += std::string("\n");
      fputs(line.c_str(), rfp);
    }

  // total line
  std::string lline =
    sformat
    ("total   : %6d %10.6f         -- %6d %10d: %10.3f + %6d %10d: %10.3f"
     " = %6d %10d %10.3f",
     tcount, terror/tcount, tfobj, tfsearch, tfsearch/double(tfobj),
     tnfobj, tnfsearch, tnfsearch/double(tnfobj),
     tobj, tsearch, tsearch/double(tobj));
  LINFO("%s", lline.c_str());
  LINFO("search count: %d: %f + %f = %f",
        itsSearchCount, double(tfobj)/itsSearchCount,
        double(tnfobj)/itsSearchCount, double(tobj)/itsSearchCount);
  lline += std::string("\n");
  fputs(lline.c_str(), rfp);
  fclose (rfp);
}

// ######################################################################
inline void GSnavResult::getSearchResult(std::vector<uint> index)
{
  uint tfObj  = 0;    uint tfSearch  = 0;
  uint tnfObj = 0;    uint tnfSearch = 0;
  uint tObj   = 0;    uint tSearch   = 0;

  // go through the indexes in question
  for(uint i = 0; i < index.size(); i++)
    {
      tfObj  += nObjectFound[index[i]];
      tnfObj += nObjectNotFound[index[i]];
      tObj   += nObject[index[i]];

      tfSearch  += nObjectFoundSearch[index[i]];
      tnfSearch += nObjectNotFoundSearch[index[i]];
      tSearch   += nObjectSearch[index[i]];
    }

  // report the result
  LINFO("%6d %10d: %10.3f + %6d %10d: %10.3f = %6d %10d %10.3f| ",
        tfObj,  tfSearch,  tfSearch/float(tfObj),
        tnfObj, tnfSearch, tnfSearch/float(tnfObj),
        tObj,   tSearch,   tSearch/float(tObj));
}

// ######################################################################

// ######################################################################

/*! This class is to read the GSnav M_result file                      */
class GSnav_M_Result
{
public:

  // input information
  std::vector<double> bbmtTime;
  double meanBbmtTime, minBbmtTime, maxBbmtTime, stdevBbmtTime;

  std::vector<double> dorsalTime;
  double meanDorsalTime, minDorsalTime, maxDorsalTime, stdevDorsalTime;

  std::vector<double> ventralTime;
  double meanVentralTime, minVentralTime, maxVentralTime, stdevVentralTime;

  std::vector<double> inputTime;
  double meanInputTime, minInputTime, maxInputTime, stdevInputTime;

  // ventral searches
  std::vector<uint> ventralSearchEndFrame;
  std::vector<uint> ventralSearchNumFrame;
  std::vector<double> ventralSearchTime;
  double meanVentralSearchTime, minVentralSearchTime,
    maxVentralSearchTime, stdevVentralSearchTime;

  // file information
  std::string savePrefix;

  inline void read(std::string inSavePrefix);
};

// ######################################################################
inline void GSnav_M_Result::read(std::string inSavePrefix)
{
  savePrefix = inSavePrefix;

  // open the result file
  std::string resFName = savePrefix + sformat("_M_results.txt");
  FILE *fp; LINFO("result file: %s", resFName.c_str());
  if((fp = fopen(resFName.c_str(),"rb")) == NULL)LFATAL("not found");

  // get each line result
  char inLine[200];
  while(fgets(inLine, 200, fp) != NULL)
    {
      uint index; float bTime, dTime, vTime, iTime;
      int rCL, rNL;  bool resetNL;

      // read the line
      sscanf(inLine, "%d %f %d %d %f %f %f",
             &index, &bTime, &rCL, &rNL, &dTime, &vTime, &iTime);
      LDEBUG("%5d %11.5f %d %d %11.5f %11.5f %11.5f",
            index, bTime, rCL, rNL, dTime, vTime, iTime);
      if(rNL > 0) resetNL = true; else resetNL = false;

      // store the results
      bbmtTime.push_back(bTime);
      dorsalTime.push_back(dTime);
      ventralTime.push_back(vTime);
      inputTime.push_back(iTime);

      // store the frames at end of ventral search
      if(resetNL)
        {
          LDEBUG("[%4" ZU "]: %d line: %s",
                 ventralSearchEndFrame.size(), index-1, inLine);

          ventralSearchEndFrame.push_back(index-1);

          if(ventralSearchNumFrame.size() == 0)
            ventralSearchNumFrame.push_back(0);
          else
            {
              uint tNum = ventralSearchNumFrame.back();
              ventralSearchNumFrame.push_back(index-1 - tNum);
            }
          ventralSearchTime.push_back(vTime);
        }
    }

  // delete the default -1.0 data entries
  bbmtTime.pop_back();
  dorsalTime.erase(dorsalTime.begin());
  ventralTime.erase(ventralTime.begin());

  ventralSearchEndFrame.erase(ventralSearchEndFrame.begin());
  ventralSearchNumFrame.erase(ventralSearchNumFrame.begin());
  ventralSearchTime.erase(ventralSearchTime.begin());

  // get the statistics (mean, min, max, stdev) for each vector
  Image<double> t1(1, bbmtTime.size(),NO_INIT);
  for(uint i = 0; i < bbmtTime.size(); i++)
    t1.setVal(0, i, double(bbmtTime[i]));
  meanBbmtTime = mean(t1);
  getMinMax(t1, minBbmtTime, maxBbmtTime);
  stdevBbmtTime = stdev(t1);

  Image<double> t2(1, dorsalTime.size(),NO_INIT);
  for(uint i = 0; i < dorsalTime.size(); i++)
    t2.setVal(0, i, double(dorsalTime[i]));
  meanDorsalTime = mean(t2);
  getMinMax(t2, minDorsalTime, maxDorsalTime);
  stdevDorsalTime = stdev(t2);

  Image<double> t3(1, ventralSearchTime.size(),NO_INIT);
  for(uint i = 0; i < ventralSearchTime.size(); i++)
    t3.setVal(0, i, double(ventralSearchTime[i]));
  meanVentralSearchTime = mean(t3);
  getMinMax(t3, minVentralSearchTime, maxVentralSearchTime);
  stdevVentralSearchTime = stdev(t3);

  Image<double> t4(1, inputTime.size(),NO_INIT);
  for(uint i = 0; i < inputTime.size(); i++)
    t4.setVal(0, i, double(inputTime[i]));
  meanInputTime = mean(t4);
  getMinMax(t4, minInputTime, maxInputTime);
  stdevInputTime = stdev(t4);
}

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
