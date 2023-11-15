/*!@file Beobot/Environment.C all the information describing
  an environment */
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
// Primary maintainer for this file: Christian Siagian <siagian@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/Environment.C $
// $Id: Environment.C 15454 2013-01-31 02:17:53Z siagian $
//

#include "Beobot/Environment.H"
#include "Image/MatrixOps.H"
#include "Util/Timer.H"

#include <sys/stat.h>
#include <errno.h>

#define LANDMARK_DB_FOLDER_NAME "LandmarkDB"

// ######################################################################
Environment::Environment(std::string envFName):
  itsLandmarkDB(new LandmarkDB()),
  itsTopologicalMap(new TopologicalMap()),
  itsSegmentRecognizer(new FeedForwardNetwork()),
  itsSegmentHistogram(new Histogram())
{
  // check if the file does not exist or it's a blank entry
  FILE *fp; if((fp = fopen(envFName.c_str(),"rb")) == NULL)
    {
      LINFO("Environment file %s not found",envFName.c_str());
      LINFO("Create a blank environment");

      // have to reset landmarkDB size by calling resetLandmarkDB

    }
  else
    {
      LINFO("Loading Environment file %s",envFName.c_str());
      load(envFName);
    }
}

// ######################################################################
bool Environment::isBlank()
{
  return (itsLandmarkDB->getNumSegment() == 0);
}

// ######################################################################
bool Environment::load(std::string fName)
{
  FILE *fp;  char inLine[200]; char comment[200];

  // open the file
  LINFO("Environment file: %s",fName.c_str());
  if((fp = fopen(fName.c_str(),"rb")) == NULL)
    { LINFO("not found"); return false; }
  std::string path = "";
  std::string::size_type spos = fName.find_last_of('/');
  if(spos != std::string::npos) path = fName.substr(0, spos+1);

  // get the number of segments
  uint nSegment;
  if (fgets(inLine, 200, fp) == NULL) LFATAL("fgets failed"); 
  sscanf(inLine, "%s %d", comment, &nSegment);
  LINFO("Segment: %d", nSegment);

  // clear and resize the landmarkDB and segment histogram
  resetNumSegment(nSegment);

  // get the map file name
  char mapcharFName[200];
  if (fgets(inLine, 200, fp) == NULL) LFATAL("fgets failed"); 
  sscanf(inLine, "%s %s", comment, mapcharFName);
  LINFO("Map file: %s", mapcharFName);
  itsTopologicalMap->read(path + std::string(mapcharFName));

  // get the segment recognizer file name
  char segRecFName[200];
  if (fgets(inLine, 200, fp) == NULL) LFATAL("fgets failed"); 
  sscanf(inLine, "%s %s", comment, segRecFName);
  LINFO("Segment Recognizer file: %s", segRecFName);
  setupSegmentRecognizer(path+std::string(segRecFName));

  // get the landmarkDB folder name
  char landmarkDBFolderName[200];
  if (fgets(inLine, 200, fp) == NULL) LFATAL("fgets failed"); 
  sscanf(inLine, "%s %s", comment, landmarkDBFolderName);
  LINFO("landmarkDB Folder Name: %s", landmarkDBFolderName);

  // skip the column headers
  if (fgets(inLine, 200, fp) == NULL) LFATAL("fgets failed");

  Timer timer(1000000);
  timer.reset();

  // populate the landmarks to the database
  uint count = 0; 
  while(fgets(inLine, 200, fp) != NULL)
    {
      int snum; char lName[200];

      // get the landmark file
      sscanf(inLine, "%d %s", &snum, lName);

      LDEBUG("[%d] landmark name: %s", snum, lName);

      // add it to the proper segment
      rutz::shared_ptr<Landmark> lmk(new Landmark(std::string(lName)));
      std::string lmkName
        (sformat("%s%s/%s", path.c_str(), landmarkDBFolderName, lName));
      //rutz::shared_ptr<Landmark> lmk(new Landmark(lmkName));
      if (!lmk->loadFrom(lmkName)) LFATAL("invalid landmark file");
      itsLandmarkDB->addLandmark(snum, lmk);
      
      LDEBUG("Added lmks[%3d][%3d]: %5d objects", snum,
             itsLandmarkDB->getNumSegLandmark(snum)-1, lmk->numObjects());

      if(count%100 == 0) 
        LINFO("loaded %4dth landmark lmks[%3d][%3d]: %5d objects", count, snum,
              itsLandmarkDB->getNumSegLandmark(snum)-1, lmk->numObjects());

      count++;
    }
  fclose(fp);
  LINFO("time: %f", timer.get()/1000.0);
  timer.reset();

  // set sessions that has the number of frames
  int ldpos = fName.find_last_of('.');
  std::string sessionFName =
    fName.substr(0, ldpos) + std::string("_sessions.txt");
  LINFO("session name: %s", sessionFName.c_str());
  itsLandmarkDB->setSession(sessionFName, false);

  // report landmark database size
  std::vector<uint> segVoCount(nSegment); uint tLmkCount = 0;
  for(uint i = 0; i < nSegment; i++)
    {
      uint nLmk = itsLandmarkDB->getNumSegLandmark(i);
      LINFO("segment[%d] has: %d landmarks ", i, nLmk);
      segVoCount[i] = 0;
      tLmkCount += nLmk;
      for(uint j = 0; j < nLmk; j++)
        {
          uint nObj = itsLandmarkDB->getLandmark(i,j)->numObjects();
          segVoCount[i] += nObj;
          LDEBUG("      lmk[%d][%d] has %d vo", i, j, nObj);
        }
    }

  uint tVoCount = 0;
  for(uint i = 0; i < nSegment; i++)
    {
      LINFO("Lmks[%d]: %d lmk (%d VO)",
            i, itsLandmarkDB->getNumSegLandmark(i), segVoCount[i]);
      tVoCount += segVoCount[i];
    }
  LINFO("%d landmarks, %d Visual Objects", tLmkCount, tVoCount);

  LINFO("time: %f", timer.get()/1000.0);

  return true;
}

// ######################################################################
void Environment::setupSegmentRecognizer(std::string gistFName)
{
  // get segment classifier parameters
  if( itsSegRecInfo.reset(gistFName.c_str()))
    {
      // instantiate a 3-layer feed-forward network
      // initialize with the provided parameters
      itsSegmentRecognizer->init3L
        (itsSegRecInfo.h1Name, itsSegRecInfo.h2Name, itsSegRecInfo.oName,
         itsSegRecInfo.redFeatSize, itsSegRecInfo.h1size,
         itsSegRecInfo.h2size, itsSegRecInfo.nOutput, 0.0, 0.0);

      // setup the PCA eigenvector
      itsPcaIcaMatrix = setupPcaIcaMatrix
        (itsSegRecInfo.trainFolder+itsSegRecInfo.evecFname,
         itsSegRecInfo.oriFeatSize, itsSegRecInfo.redFeatSize);
    }
  else
    LINFO("Segment classifier file: %s not found", gistFName.c_str());
}

// ######################################################################
void Environment::resetNumSegment(uint nSegment)
{
  itsLandmarkDB->resize(nSegment);
  itsSegmentHistogram->resize(nSegment);
}

// ######################################################################
void Environment::setCurrentSegment(uint currSegNum)
{
  ASSERT(currSegNum < uint(itsSegmentHistogram->getSize()));
  itsSegmentHistogram->clear();
  itsSegmentHistogram->addValue(currSegNum, 1.0);
  itsCurrentSegment = currSegNum;
}

// ######################################################################
bool Environment::save
(std::string fName, std::string tfName, std::string envPrefix)
{
  bool ts = save(itsTempLandmarkDB, tfName, envPrefix);
  bool s  = save(itsLandmarkDB, fName, envPrefix);
  return (ts && s);
}

// ######################################################################
bool Environment::save(std::string fName, std::string envPrefix)
{
  return save(itsLandmarkDB, fName, envPrefix);
}

// ######################################################################
bool Environment::save(rutz::shared_ptr<LandmarkDB> landmarkDB,
                       std::string fName, std::string envPrefix)
{
  // open an environment file
  FILE * envFile = fopen(fName.c_str(), "wt");
  int nSegment = landmarkDB->getNumSegment();

  if (envFile!=NULL)
    {
      LINFO("saving environment to %s", fName.c_str());

      // get the directory name - include the "/"
      int lspos = fName.find_last_of('/');
      int ldpos = fName.find_last_of('.');
      std::string filepathstr = fName.substr(0, lspos+1);
      LINFO("path: %s", filepathstr.c_str());

      std::string lmfprefix =
        fName.substr(lspos+1, ldpos - lspos - 1);
      LINFO("prefix: %s", lmfprefix.c_str());

      std::string segnum(sformat("segment#: %d\n",nSegment));
      fputs (segnum.c_str(), envFile);

      std::string mapFileStr =
        envPrefix + std::string(".tmap");
      std::string mapInfo(sformat("mapFile: %s\n", mapFileStr.c_str()));
      fputs (mapInfo.c_str(), envFile);

      std::string gistFileStr =
        envPrefix + std::string("_GIST_train.txt");
      std::string gistInfo(sformat("gistFile: %s\n", gistFileStr.c_str()));
      fputs (gistInfo.c_str(), envFile);

      // create a landmarkDB folder
      std::string landmarkDBfolder =
        filepathstr + LANDMARK_DB_FOLDER_NAME;
      if (mkdir(landmarkDBfolder.c_str(), 0777) == -1 && errno != EEXIST)
        {
          LFATAL("Cannot create landmarkDB folder: %s",
                 landmarkDBfolder.c_str());
        }
      LINFO("creating landmarkDB folder: %s", landmarkDBfolder.c_str());

      std::string landmarkDBfolderInfo
        (sformat("LandmarkDBFolderName: %s\n", LANDMARK_DB_FOLDER_NAME));
      fputs (landmarkDBfolderInfo.c_str(), envFile);

      std::string header("Segment        Landmark Name\n");
      fputs (header.c_str(), envFile);

      // re-create if we are adding from the previous one
      for(int i = 0; i < nSegment; i++)
        {
          for(uint j = 0; j < landmarkDB->getNumSegLandmark(i); j++)
            {
              // file name (no .lmk extension)
              std::string lmkName
                (sformat("%s_%03d_%03d", lmfprefix.c_str(), i, j));
              LDEBUG("landmark name: %s", lmkName.c_str());
              landmarkDB->getLandmark(i,j)->setName(lmkName);

              // full name (path + file name + .lmk)
              std::string lmfName =
                landmarkDBfolder +  std::string("/") +
                lmkName + std::string(".lmk");
              LDEBUG("%s", lmfName.c_str());

              // save the info and write the name
              landmarkDB->getLandmark(i,j)->saveTo(lmfName);
              fputs (sformat("%-14d %s.lmk\n", i, lmkName.c_str()).c_str(),
                     envFile);
            }
        }
      fclose (envFile);
    }
  else return false;

  return true;
}

// ######################################################################
Environment::~Environment()
{
}

// ######################################################################
rutz::shared_ptr<Histogram> Environment::classifySegNum(Image<double> cgist)
{
  // reduce feature dimension (if available)
  Image<double> in;
  if(itsSegRecInfo.isPCA) in = matrixMult(itsPcaIcaMatrix, cgist);
  else in = cgist;

  // analyze the gist features to recognize the segment
  Image<double> res = itsSegmentRecognizer->run3L(in);

  itsSegmentHistogram->clear();

  float vMax = 0.0; int segMax = 0;
  for(int i = 0; i < itsSegmentHistogram->getSize(); i++)
    {
      float cVal = res[Point2D<int>(0,i)];
      LDEBUG("seg[%3d]: %.4f",i, cVal);
      if(cVal > vMax){ vMax = cVal; segMax = i; }

      itsSegmentHistogram->addValue(i, cVal);
    }
  LDEBUG("predicted segment number %d: %f", segMax, vMax);
  itsCurrentSegment = segMax;

  return itsSegmentHistogram;
}

// ######################################################################
void Environment::startBuild()
{
  // to store the currently build landmarkDB
  itsTempLandmarkDB.reset(new LandmarkDB(itsLandmarkDB->getNumSegment()));
  itsTempLandmarkDB->setWindow(itsWin);
}

// ######################################################################
void Environment::build
( std::vector<rutz::shared_ptr<VisualObject> > &inputVO,
  std::vector<Point2D<int> > &objOffset, uint fNum,
  rutz::shared_ptr<VisualObject> scene)
{
  // change folder name to be the landmark folder name
  for(uint i = 0; i < inputVO.size(); i++) 
    {
      std::string tfname = inputVO[i]->getImageFname().c_str();
      std::string::size_type spos =  tfname.find_last_of('/');	 
      std::string tname1("");
      std::string tname2 = tfname; 
      if(spos != std::string::npos)	 
        { 
          tname1 = tfname.substr(0,spos+1); 
          tname2 = tfname.substr(spos+1); 
        }
      std:: string tname = tname1 + LANDMARK_DB_FOLDER_NAME + "/" + tname2;
      inputVO[i]->setImageFname(tname);
    }

  // build the landmarkDB
  itsTempLandmarkDB->build
    (inputVO, objOffset, fNum, itsCurrentSegment, scene);
}

// ######################################################################
void Environment::finishBuild
(std::string envFName, std::string sessionPrefix, uint rframe)
{
  itsTempLandmarkDB->finishBuild(rframe);
  //itsTempLandmarkDB->display();

  // combine with the actual landmark Database
  LINFO("Combine landmarkDB & tempLandmarkDB");
  Raster::waitForKey();
  itsLandmarkDB = combineLandmarkDBs(itsLandmarkDB, itsTempLandmarkDB);
  itsLandmarkDB->setWindow(itsWin);

  // get the directory name - include the "/"
  int lspos = envFName.find_last_of('/');
  int ldpos = envFName.find_last_of('.');
  std::string path = envFName.substr(0, lspos+1);
  LINFO("path: %s", path.c_str());

  std::string envPrefix = envFName.substr(lspos+1, ldpos - lspos - 1);
  LINFO("prefix: %s", envPrefix.c_str());

  // append the session to the session list
  std::string sessionFName = path + envPrefix + sformat("_sessions.txt");
  FILE *rFile = fopen(sessionFName.c_str(), "at");
  if (rFile != NULL)
    {
      LINFO("appending session to %s", sessionFName.c_str());
      std::string line =
        sformat("%-20s  %-10d %-10d %-10d\n",
                sessionPrefix.c_str(), 0, rframe, itsCurrentSegment);
      fputs(line.c_str(), rFile);
      fclose (rFile);
    }
  else LINFO("can't create file: %s", sessionFName.c_str());

  Raster::waitForKey();

  // reset the session related information and sort the landmarks as well
  itsLandmarkDB->setSession(sessionFName, true);
  //itsLandmarkDB->display();
}

// ######################################################################
rutz::shared_ptr<LandmarkDB> Environment::combineLandmarkDBs
( rutz::shared_ptr<LandmarkDB> lmks1, rutz::shared_ptr<LandmarkDB> lmks2)
{
  // NOTE: we are NOT creating DEEP COPY of landmarks
  uint nSegment = lmks1->getNumSegment();
  rutz::shared_ptr<LandmarkDB> resLmks(new LandmarkDB(nSegment));

  // go through all segments
  for(uint  i= 0; i < nSegment; i++)
    {
      uint nlmk1 = lmks1->getNumSegLandmark(i);
      uint nlmk2 = lmks2->getNumSegLandmark(i);
      LINFO("segment %d: lmks1[]: %d lmks2[]: %d",i, nlmk1, nlmk2);

      // if both lmks1 and lmks2 has no landmarks
      // it will go through without consequences

      // if only lmks2 has no landmarks in this segment
      if(nlmk1 > 0 && nlmk2 == 0)
        {
           LINFO("Lmks2[%d] is empty; add all Lmks1 landmarks", i);

          // just push in all the landmarks in lmks1
          for(uint j = 0; j < nlmk1; j++)
            {
              resLmks->addLandmark(i,lmks1->getLandmark(i,j));
            }
        }
      // if only lmks1 has no landmarks in this segment
      else if(nlmk1 == 0 && nlmk2 > 0)
        {
           LINFO("Lmks1[%d] is empty; add all Lmks2 landmarks", i);

          // just push in all the landmarks in lmks2
          for(uint j = 0; j < nlmk2; j++)
            {
              resLmks->addLandmark(i,lmks2->getLandmark(i,j));
            }
        }
      // else we have some combining to do
      else
        {
          // match lmk2 with lmk1

          // create a temporary storage for each landmark in lmks1
          // indicating whether they have been added in or not
          std::vector<bool> lmks1added(nlmk1);
          std::vector<bool> lmks1appended(nlmk1);
          std::vector<int>  lmks1pos  (nlmk1);
          for(uint j = 0; j < nlmk1; j++)
            {
              lmks1added[j]    = false;
              lmks1appended[j] = false;
              lmks1pos  [j]    = -1;
            }

          // for each of the landmarks in the lmks2
          for(uint j = 0; j < nlmk2; j++)
            {
              LINFO("checking lmks2[%d][%d]", i, j);

              // for each of the landmark in lmks1
              std::vector<uint> mIndex;
              for(uint k = 0; k < nlmk1; k++)
                {
                  if(!lmks1appended[k])
                    {
                      // check for number of matches of objects in landmarks
                      // relax the distance of the sal point to 15.0 pixels
                      uint nMatch =
                        lmks1->getLandmark(i,k)->numMatch
                        (lmks2->getLandmark(i,j), 15.0F);
                      uint nObj1 = lmks1->getLandmark(i,k)->numObjects();
                      uint nObj2 = lmks2->getLandmark(i,j)->numObjects();
                      float mPntg = float(nMatch)/nObj2;

                      // check if match goes pass any of the thresholds:
                      // nmatches <= 5, pntg >= 50%
                      // nmatches 5 < x <= 10 pntg >= 25%
                      // nmatches > 10
                      bool combine = false;
                      if((nMatch > 1  && nMatch <=  5 && mPntg >= .50) ||
                         (nMatch > 5  && nMatch <= 10 && mPntg >= .25) ||
                         (nMatch > 10))
                        {
                          combine = true; mIndex.push_back(k);
                          LINFO("<%d> lmks1[%d][%d](%d)-lmks2[%d][%d](%d)="
                                " %d/%d = %f", combine,
                                i, k, nObj1, i, j, nObj2, nMatch, nObj2, mPntg);
                          //if(combine) Raster::waitForKey();
                        }
                    }
                }

              // if we don't have any matches for lmks2[j]
              if(mIndex.size() == 0)
                {
                  LINFO("match not found - adding Lmks2[%d]", j);

                  // add it to resulting DB: Not DEEP copy
                  resLmks->addLandmark(i, lmks2->getLandmark(i,j));
                  //Raster::waitForKey();
                }
               else
                {
                  // if > 1 lmks2[j] matches
                  // then we have to append all the matches first
                  for(uint k = 1; k < mIndex.size(); k++)
                    {
                      // if the landmark of lmks1 is not added yet
                      if(!lmks1added[mIndex[0]])
                        {
                          // create a blank one to add both entries
                          rutz::shared_ptr<Landmark> resLmk(new Landmark());
                          std::string resLmkName
                            (sformat("Comb1_%d-1_%d", mIndex[0], mIndex[k]));
                          resLmk->setName(resLmkName);
                          resLmk->setMatchWin
                            (lmks1->getLandmark(i,mIndex[0])->getMatchWin());
                          resLmk->combine(lmks1->getLandmark(i,mIndex[0]),
                                          lmks1->getLandmark(i,mIndex[k]));

                          // add the combination to the result landmark DB
                          resLmks->addLandmark(i, resLmk);

                          // note that lmks1[mIndex] is in the result lmks
                          lmks1added[mIndex[0]] = true;
                          lmks1pos  [mIndex[0]] =
                            resLmks->getNumSegLandmark(i) - 1;

                          LINFO("NEW COMBO Landmark: %s", resLmkName.c_str());
                        }
                      else
                        {
                          // otherwise append it to the end of the combined lmk
                          std::string resLmkName =
                            resLmks->getLandmark(i,lmks1pos[mIndex[0]])->
                            getName()+std::string(sformat(",1_%d", mIndex[k]));
                          resLmks->getLandmark(i,lmks1pos[mIndex[0]])->
                            setName(resLmkName);
                          resLmks->getLandmark(i,lmks1pos[mIndex[0]])->
                            append(lmks1->getLandmark(i,mIndex[k]));

                          LINFO("APP COMBO Landmark: %s",resLmkName.c_str());
                        }

                      lmks1added[mIndex[k]] = true;
                      lmks1pos  [mIndex[k]] = resLmks->getNumSegLandmark(i)-1;
                      lmks1appended[mIndex[k]] = true;
                      LINFO("append: lmks1[%d & %d]", mIndex[0], mIndex[k]);
                    }

                  // combining lmks2[j] and lmks1[mIndex[0]]
                  LINFO("Combolmk:[lmks1[%d] - lmks2[%d]]", mIndex[0], j);
                  if(!lmks1added[mIndex[0]])
                    {
                      rutz::shared_ptr<Landmark> resLmk(new Landmark());
                      std::string resLmkName
                        (sformat("Comb1_%d-2_%d", mIndex[0], j));
                      resLmk->setName(resLmkName);
                      resLmk->setMatchWin(lmks1->getLandmark(i,mIndex[0])->
                                          getMatchWin());
                      resLmk->combine
                        (lmks1->getLandmark(i,mIndex[0]),
                         lmks2->getLandmark(i,j));

                      // add the combination to the result landmark DB
                      resLmks->addLandmark(i, resLmk);

                      // note that lmks1[mIndex[0]] is in the result lmks
                      lmks1added[mIndex[0]] = true;
                      lmks1pos  [mIndex[0]] = resLmks->getNumSegLandmark(i)-1;

                      LINFO("NEW COMB Landmark: %s", resLmkName.c_str());
                    }
                  else
                    {
                      // otherwise append it to the end of the combined lmk
                      std::string resLmkName =
                        resLmks->getLandmark(i,lmks1pos[mIndex[0]])->getName()
                        + std::string(sformat(",2_%d",j));
                      resLmks->getLandmark(i,lmks1pos[mIndex[0]])->
                        setName(resLmkName);
                      resLmks->getLandmark(i,lmks1pos[mIndex[0]])->
                        append(lmks2->getLandmark(i,j));

                      LINFO("APP COMBO Landmark: %s", resLmkName.c_str());
                    }
                  //Raster::waitForKey();
                }
            }

          // add the rest of the landmarks in lmks1 to the resulting lmks
          for(uint k = 0; k < nlmk1; k++)
            {
              if(!lmks1added[k])
                {
                  LINFO("adding lmks1[%d][%d] to the reslmks", i, k);
                  resLmks->addLandmark(i,lmks1->getLandmark(i,k));
                }
            }
        }
    }

  LINFO("done combining");
  for(uint  i= 0; i < nSegment; i++)
    {
      for(uint j = 0; j < resLmks->getNumSegLandmark(i); j++)
        {
          LINFO("res[%d][%d]: %s",i,j,
                resLmks->getLandmark(i,j)->getName().c_str());
        }
    }

  Raster::waitForKey();
  return resLmks;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
