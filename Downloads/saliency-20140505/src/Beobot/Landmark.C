/*!@file Beobot/Landmark.C Landmark class for localization */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/Landmark.C $
// $Id: Landmark.C 15441 2012-11-14 21:28:03Z kai $
//
// FIXXX ADD: motion recognition
//      different VOMA

#include "Beobot/Landmark.H"
#include "Util/Timer.H"
#include "Raster/Raster.H"
#include "Image/CutPaste.H"
#include "Image/DrawOps.H"
#include "Image/MathOps.H"

#include <iostream>
#include <list>
#include <fstream>

#define FWAIT_NUM                    5U
#define MIN_LINKS                    3U
#define NUM_FRAME_LOOKBACK           10

// ######################################################################
Landmark::Landmark(const std::string& name):
  itsName(name),
  itsVoma (new VisualObjectMatchAlgo(VOMA_SIMPLE)),
  //VOMA_KDTREE or VOMA_KDTREEBBF
  itsVisualObjectDB(new VisualObjectDB()),
  itsOffsetCoords(),
  itsVisualObjectFNum(),
  itsTempVisualObjectDB(new VisualObjectDB()),
  itsTempOffsetCoords(),
  itsTempVisualObjectFNum(),
  itsKPtrackers(),
  itsCip()
{
  // these numbers are default value (may be incorrect)
  // should check if there are objects in the DB first
  itsLatestVisualObjectFNum      = 0;
  itsLatestTempVisualObjectFNum  = 0;
}

// ######################################################################
Landmark::Landmark(rutz::shared_ptr<VisualObject>& obj,
                   Point2D<int> objOffset, uint fNum,
                   const std::string& name):
  itsName(name),
  itsVoma (new VisualObjectMatchAlgo(VOMA_SIMPLE)),
  //VOMA_KDTREE or VOMA_KDTREEBBF
  itsVisualObjectDB(new VisualObjectDB()),
  itsOffsetCoords(),
  itsVisualObjectFNum(),
  itsTempVisualObjectDB(new VisualObjectDB()),
  itsTempOffsetCoords(),
  itsTempVisualObjectFNum(),
  itsKPtrackers(),
  itsCip()
{
  itsLatestVisualObjectFNum      = fNum;
  itsLatestTempVisualObjectFNum  = 0;
  init(obj, objOffset, fNum);
}

// ######################################################################
void Landmark::init(rutz::shared_ptr<VisualObject>& obj,
                    Point2D<int> objOffset, uint fNum)
{
  // add the first object to the newly created databases
  // landmarks are found because it is salient
  // or labelled salient by the input program
  addObject(obj, objOffset, fNum);

  // register the keypoints locations
  for(uint i = 0; i < obj->numKeypoints(); i++)
    {
      rutz::shared_ptr<KeypointTracker>
        newKPtracker(new KeypointTracker(sformat("KPtrk%d_%d", fNum, i)));
      newKPtracker->add(obj->getKeypoint(i), objOffset, fNum);
      itsKPtrackers.push_back(newKPtracker);
    }
  LINFO("there are %" ZU " keypoints tracked", itsKPtrackers.size());
}

// ######################################################################
bool Landmark::loadFrom(const std::string& fname)
{
  // load the actual VisualObjectDatabase
  bool succeed = false;  

  // FILE *fp; 
  // if((fp = fopen(fname.c_str(),"rb")) == NULL)
  //   LFATAL("Landmark file %s not found", fname.c_str());  
  // LINFO("Loading Environment file %s",envFName.c_str());

  const char *fn = fname.c_str();
  LDEBUG("Loading Visual Object database: '%s'...", fn);

  std::ifstream inf(fn);
  if (inf.is_open() == false) 
    { LERROR("Cannot open '%s' -- USING EMPTY", fn); succeed = false; }
  else
    {
      // inf>>(*this);
      std::string name;
      std::getline(inf, name);
      
      itsVisualObjectDB->setName(name);
      
      uint size; inf>>size;      
      itsVisualObjectDB->clearObjects(size); 
      
      uint count = 0;
      while (count < size)
        {
          rutz::shared_ptr<VisualObject> newvo(new VisualObject());

          //is>>(*newvo);
          newvo->createVisualObject(inf, *newvo, false);

          std::string iname = newvo->getImageFname();

          // we add the actual path to the image file name	 
          std::string::size_type spos =  fname.find_last_of('/');	 
          std::string filePath("");
          if(spos != std::string::npos)	 
            filePath = fname.substr(0,spos+1);	          

          // make sure file path are stripped from the image name
          std::string::size_type spos2 =  iname.find_last_of('/');	 
          if(spos2 != std::string::npos) iname = iname.substr(spos2+1);

          std::string fiName =  filePath + iname;          
          newvo->setImageFname(fiName);          
          newvo->loadImage();

          itsVisualObjectDB->setObject(count, newvo);
          count++;
          LDEBUG("[%d] %s ",  count, fiName.c_str());
        }

      inf.close();
      LDEBUG("Done. Loaded %u VisualObjects.", numObjects());
      succeed = true;
    }
  //Raster::waitForKey();
  
  //if(itsVisualObjectDB->loadFrom(fname,false))
  if(succeed)
    {
      // set the name
      itsName = itsVisualObjectDB->getName();

      // extract the other information from the name
      // I know. it's clever
      for(uint i = 0; i < itsVisualObjectDB->numObjects(); i++)
        {
          std::string tName = itsVisualObjectDB->getObject(i)->getName();
          int ioff, joff, fNum;
          std::string temp;

          // ObjName_Ioffset_Ioffset_fNum
          // we parse it in reverse
          LDEBUG("tName: %s", tName.c_str());
          int lupos = tName.find_last_of('_');
          temp = tName.substr(lupos+1, tName.length()-lupos-1);
          tName = tName.substr(0, lupos);
          fNum = atoi(temp.c_str());

          lupos = tName.find_last_of('_');
          temp = tName.substr(lupos+1, tName.length()-lupos-1);
          tName = tName.substr(0, lupos);
          joff = atoi(temp.c_str());

          lupos = tName.find_last_of('_');
          temp = tName.substr(lupos+1, tName.length()-lupos-1);
          tName = tName.substr(0, lupos);
          ioff = atoi(temp.c_str());

          tName = tName.substr(0, lupos);

          LDEBUG("[%3d] %30s: (%3d %3d) fnum: %5d",
                 i, tName.c_str(), ioff, joff, fNum);

          itsOffsetCoords.push_back(Point2D<int>(ioff, joff));
          itsVisualObjectFNum.push_back(fNum);

          itsVisualObjectDB->getObject(i)->setName(tName);

        }
      return true;
    }
  else return false;
}

// ######################################################################
void Landmark::setSessionInfo()
{
  itsSession.clear();
  itsSessionIndexRange.clear();
  ASSERT(itsVisualObjectDB->numObjects() > 0);

  std::string lname; uint sind = 0, eind = 0;
  for(uint i = 0; i < itsVisualObjectDB->numObjects(); i++)
    {
      // get the proper session stem
      // take out the suffix object numbers: _SAL_xxxxxxx_xx
      std::string sname = itsVisualObjectDB->getObject(i)->getName();
      sname = sname.substr(0, sname.find_last_of('_'));
      sname = sname.substr(0, sname.find_last_of('_'));
      sname = sname.substr(0, sname.find_last_of('_'));

      // if session name changed then add it to the list
      if(i != 0 && sname.compare(lname))
        {
          itsSessionIndexRange.push_back(std::pair<uint,uint>(sind,eind));
          itsSession.push_back(lname);
          sind = i;
        }
      lname = sname;
      eind = i;
    }

  // add the last session if needed
  itsSessionIndexRange.push_back(std::pair<uint,uint>(sind,eind));
  itsSession.push_back(lname);

  // print the sessions
  uint nSession = itsSession.size();
  for(uint i = 0; i < nSession; i++)
    LDEBUG("session[%3d] %30s: [%4d %4d]", i, itsSession[i].c_str(),
           itsSessionIndexRange[i].first, itsSessionIndexRange[i].second);

  // compute the average salient features of the VO in the landmark
  computeSalientFeatures();
}

// ######################################################################
void Landmark::computeSalientFeatures()
{
  uint nObject = numObjects();
  if (nObject == 0) return;
  uint nFeatures = itsVisualObjectDB->getObject(0)->numFeatures();
  itsSalientFeatures.resize(nFeatures);

  // get features of each object
  for(uint i = 0; i < nFeatures; i++)
    {
      Image<double> temp(1,nObject, NO_INIT);
      Image<double>::iterator aptr = temp.beginw(), stop = temp.endw();

      uint j = 0;
      while (aptr != stop)
        {
          *aptr++ = itsVisualObjectDB->getObject(j)->getFeature(i); j++;
        }
      double tMean  = mean(temp);
      double tStdev = stdev(temp);
      LDEBUG("[%5d] m: %f s: %f", i, tMean, tStdev);

      itsSalientFeatures[i] = std::pair<double,double>(tMean, tStdev);
    }
}

// ######################################################################
float Landmark::matchSalientFeatures(rutz::shared_ptr<VisualObject> object)
{
//  uint nObject = numObjects();
//  if (nObject == 0) return -1.0;

  std::vector<float> objFeat = object->getFeatures();

  // check with every objects
//  float min = 1.0; float max = 0.0;
//   for(uint j = 0; j < nObject; j++)
//     {
//       std::vector<double> feat2 = getObject(j)->getFeatures();

//       // feature similarity  [ 0.0 ... 1.0 ]:
//       float cval = 0.0;
//       for(uint i = 0; i < objFeat.size(); i++)
//         {
//           float val = pow(objFeat[i] - feat2[i], 2.0); cval += val;
//         }
//       cval = sqrt(cval/objFeat.size());
//       if(cval < min) min = cval;
//       if(cval > max) max = cval;
//       LDEBUG("cval[%d]: %f", j, cval);
//     }

  // feature similarity  [ 0.0 ... 1.0 ]:
  float cval = 0.0;

  std::vector<float>::iterator aptr = objFeat.begin(), stop = objFeat.end();
  std::vector<std::pair<double,double> >::iterator
    bptr = itsSalientFeatures.begin();
  while (aptr != stop)
    {
      float val = pow(*aptr++ - (*bptr).first, 2.0F); bptr++;
      cval += val;
    }

//   float cval2 = 0.0F;
//   for(uint i = 0; i < objFeat.size(); i++)
//     {
//       float val = pow(objFeat[i] - itsSalientFeatures[i].first, 2.0F);
//       cval2 += val;
//     }

  cval = sqrt(cval/objFeat.size());
//  LDEBUG("min: %f: max: %f cval: %f", min, max, cval);

  // range: [ 0.0 ... 1.0 ] the lower the better the match
  return cval;
}

// ######################################################################
bool Landmark::saveTo(const std::string& fname)
{
  // put the Landmark name as the DB name
  itsVisualObjectDB->setName(itsName);

  uint nObjects = itsVisualObjectDB->numObjects();
  std::vector<std::string> tNames(nObjects);

  // put all the individual object information on the VO name
  // ObjName iOffset jOffset fNum
  for(uint i = 0; i < nObjects; i ++)
    {
      int ioff, joff, fNum;
      ioff = itsOffsetCoords[i].i;
      joff = itsOffsetCoords[i].j;
      fNum = itsVisualObjectFNum[i];

      std::string orgName = itsVisualObjectDB->getObject(i)->getName();
      tNames[i] = orgName;
      LDEBUG("%s__%d_%d_%d", orgName.c_str(), ioff, joff, fNum);
      std::string
        appName(sformat("%s_%d_%d_%d",
                        orgName.c_str(), ioff, joff, fNum));

      itsVisualObjectDB->getObject(i)->setName(std::string(appName));
    }

  // save the VisualObject DB
  bool retVal = itsVisualObjectDB->saveTo(fname);

  // revert back to the original names
  for(uint i = 0; i < nObjects; i ++)
    {
      itsVisualObjectDB->getObject(i)->setName(tNames[i]);
    }
  return retVal;
}

// ######################################################################
Landmark::~Landmark()
{ }

// ######################################################################
// build the landmark by adding the new Visual Object
rutz::shared_ptr<VisualObjectMatch>
Landmark::build(rutz::shared_ptr<VisualObject> obj,
                Point2D<int> objOffset, uint fNum)
{
  // if this is the first item
  if(numObjects() == 0)
    {
      init(obj, objOffset, fNum);
      return rutz::shared_ptr<VisualObjectMatch>();
    }
  else
    return input(obj, objOffset, fNum);
}

// ######################################################################
// build the landmark by adding visual object with no salient point
// (image of whole frame, for example)
rutz::shared_ptr<VisualObjectMatch>
Landmark::build(rutz::shared_ptr<VisualObject> obj, uint fNum)
{
  return build(obj, Point2D<int>( 0,  0), fNum);
}

// ######################################################################
// just like build except it does not add the object
// only return where it would have been placed (in DB or Temp DB)
// or if it is rejected altogether
rutz::shared_ptr<VisualObjectMatch>
Landmark::buildCheck(rutz::shared_ptr<VisualObject> obj,
                     Point2D<int> objOffset, uint fNum,
                     bool &inDB, bool &inTDB, int &tIndex)
{
  // if this is the first item
  if(numObjects() == 0)
    {
      // init situation
      inDB = false; inTDB = false;
      return rutz::shared_ptr<VisualObjectMatch>();
    }
  else
    return inputCheck(obj, objOffset, fNum, inDB, inTDB, tIndex);
}

// ######################################################################
// corresponding non salient object input
rutz::shared_ptr<VisualObjectMatch>
Landmark::buildCheck(rutz::shared_ptr<VisualObject> obj, uint fNum,
                bool &inDB, bool &inTDB, int &tIndex)
{
  return buildCheck(obj, Point2D<int>( 0,  0), fNum,
                    inDB, inTDB, tIndex);
}

// ######################################################################
// build the landmark using salient object
// based on information passed by build check
rutz::shared_ptr<VisualObjectMatch>
Landmark::build(rutz::shared_ptr<VisualObject> obj,
                Point2D<int> objOffset, uint fNum,
                bool inDB, bool inTDB, int tIndex,
                rutz::shared_ptr<VisualObjectMatch> cmatch)
{
  // if this is the first item
  if(numObjects() == 0)
    {
      init(obj, objOffset, fNum);
      return rutz::shared_ptr<VisualObjectMatch>();
    }
  else
    return input(obj, objOffset, fNum, inDB, inTDB, tIndex, cmatch);
}

// ######################################################################
// build the landmark using non-salient object
// based on information passed by build check
rutz::shared_ptr<VisualObjectMatch>
Landmark::build(rutz::shared_ptr<VisualObject> obj, uint fNum,
                bool inDB, bool inTDB, int tIndex,
                rutz::shared_ptr<VisualObjectMatch> cmatch)
{
  return build(obj, Point2D<int>(0,0), fNum, inDB, inTDB, tIndex, cmatch);
}

// ######################################################################
// track the landmark using the new Visual Object
rutz::shared_ptr<VisualObjectMatch>
Landmark::input(rutz::shared_ptr<VisualObject> obj,
                Point2D<int> objOffset, uint fNum)
{
  Image<PixRGB<byte> > objImg = obj->getImage();
  int objW = objImg.getDims().w();
  int objH = objImg.getDims().h();

  Rectangle objRect(objOffset, objImg.getDims());
  LINFO("%s (sal:[%d %d]) info: at [%d, %d, %d, %d]: %d by %d",
        obj->getName().c_str(), obj->getSalPoint().i, obj->getSalPoint().j,
        objOffset.i, objOffset.j,
        objOffset.i+objW-1, objOffset.j + objH-1, objW, objH);

  // find match with the last NUM_FRAME_LOOKBACK
  //  objects in the DB
  rutz::shared_ptr<VisualObjectMatch> cmatch;
  int mInd = findDBmatch(obj, cmatch, NUM_FRAME_LOOKBACK);
  LINFO("found DB match: %d", mInd);

  // if match not found and DB is empty
  if(mInd == -1)
    {
      int mtInd = findTempDBmatch(obj, cmatch, NUM_FRAME_LOOKBACK);
      LINFO("found TDB match: %d",mtInd);

      // if we find a match with a temp object
      if(mtInd != -1)
        {
          // move the temp object to DB
          moveTempVisualObjectToDB(mtInd);
          LINFO("now have %d objects",
                itsVisualObjectDB->numObjects());
        }
      else
        {
          // also could not find a match in the temp holding
          LINFO("not adding %s (%d) to landmark %s",
                obj->getName().c_str(), fNum, itsName.c_str());
          return cmatch;
        }
    }

  // special framing for checking with whole scene
  if(obj->getSalPoint().i == -1 && obj->getSalPoint().j == -1)
    cmatch = cropInputImage(obj, objOffset, mInd, cmatch);

  // add object to the temp db:
  LINFO("put to temp holding first");
  tAddObject(obj, objOffset, fNum);

  // use the matches from the actual inserted object
  //trackKeypoints(cmatch, mInd);

  return cmatch;
}

// ######################################################################
// track check the landmark using the new Visual Object
rutz::shared_ptr<VisualObjectMatch>
Landmark::inputCheck(rutz::shared_ptr<VisualObject> obj,
                     Point2D<int> objOffset, uint fNum,
                     bool &inDB, bool &inTDB, int &tIndex)
{
  Image<PixRGB<byte> > objImg = obj->getImage();
  int objW = objImg.getDims().w();
  int objH = objImg.getDims().h();

  Rectangle objRect(objOffset, objImg.getDims());
  LINFO("%s (sal:[%d %d]) info: at [%d, %d, %d, %d]: %d by %d",
        obj->getName().c_str(), obj->getSalPoint().i, obj->getSalPoint().j,
        objOffset.i, objOffset.j,
        objOffset.i+objW-1, objOffset.j + objH-1, objW, objH);

  // find match with the last NUM_FRAME_LOOKBACK
  // objects in the DB
  rutz::shared_ptr<VisualObjectMatch> cmatch;
  int mInd = findDBmatch(obj, cmatch, NUM_FRAME_LOOKBACK);
  LINFO("find DB match: %d", mInd);

  // if match not found and DB is empty
  if(mInd == -1)
    {
      inDB = false;
      int mtInd = findTempDBmatch(obj, cmatch, NUM_FRAME_LOOKBACK);
      LINFO("found match at temp: %d",mtInd);

      // if we find a match with a temp objects
      if(mtInd != -1) { inTDB = true; tIndex = mtInd; }
      else
        {
          // also could not find a match in the temp holding
          inTDB = false; tIndex = -1;
          LINFO("cannot add %s (%d) to landmark %s",
                obj->getName().c_str(), fNum, itsName.c_str());
          return cmatch;
        }
    }
  // it matches with DB, so TDB is irrelevant (default to false)
  else { inDB = true; inTDB = false; tIndex = mInd; }

  return cmatch;
}

// ######################################################################
// track the landmark using the new Visual Object
// with the information passed by buildcheck
rutz::shared_ptr<VisualObjectMatch>
Landmark::input(rutz::shared_ptr<VisualObject> obj,
                Point2D<int> objOffset, uint fNum,
                bool inDB, bool inTDB, int tIndex,
                rutz::shared_ptr<VisualObjectMatch> cmatch)
{
  Image<PixRGB<byte> > objImg = obj->getImage();
  int objW = objImg.getDims().w();
  int objH = objImg.getDims().h();

  Rectangle objRect(objOffset, objImg.getDims());
  LINFO("%s (sal:[%d %d]) at [%d, %d, %d, %d]: %d by %d: {%d %d %d}",
        obj->getName().c_str(), obj->getSalPoint().i, obj->getSalPoint().j,
        objOffset.i, objOffset.j,
        objOffset.i+objW-1, objOffset.j + objH-1, objW, objH,
        inDB, inTDB,tIndex);

  // if not match just return without inserting
  if(!inDB && !inTDB)
    {
      // no matching was done, return null pointer
      return rutz::shared_ptr<VisualObjectMatch>();
    }

  // if match was found at TDB
  if(!inDB && inTDB)
    {
      // move the temp object to DB
      moveTempVisualObjectToDB(tIndex);
      LINFO("now have %d objects", itsVisualObjectDB->numObjects());
    }

  // special framing for checking with whole scene
  if(obj->getSalPoint().i == -1 && obj->getSalPoint().j == -1)
    cmatch = cropInputImage(obj, objOffset, tIndex, cmatch);

  // add object to the temp db:
  LINFO("put to temp holding first");
  tAddObject(obj, objOffset, fNum);

  // no matching was done, return null pointer
  return rutz::shared_ptr<VisualObjectMatch>();
}

// ######################################################################
rutz::shared_ptr<VisualObjectMatch> Landmark::cropInputImage
( rutz::shared_ptr<VisualObject> &obj, Point2D<int> &objOffset,
  int mInd, rutz::shared_ptr<VisualObjectMatch> cmatch)
{
  const Image<PixRGB<byte> > objImg = obj->getImage();

  // get the bounding box for tracked subregion
  Rectangle tempR = cmatch->getOverlapRect();
  LINFO("getOverlapRect [ %d, %d, %d, %d ]",
        tempR.top(), tempR.left(), tempR.bottomI(), tempR.rightI());
  tempR = tempR.getOverlap(objImg.getBounds());

  // if we are tracking,
  // make sure the image does not increase uncontrollably
  Dims prevDims =
    itsVisualObjectDB->getObject(mInd)->getImage().getDims();
  if((tempR.width()  > 1.05 * prevDims.w()) ||
     (tempR.height() > 1.05 * prevDims.h())   )
    {
      LINFO("%d by %d  vs %d by %d resize window manually",
            tempR.width(),tempR.height(), prevDims.w(), prevDims.h());

      Point2D<int> nPt(int(-cmatch->getSIFTaffine().tx),
                  int(-cmatch->getSIFTaffine().ty) );

      tempR = Rectangle(nPt, prevDims).getOverlap(objImg.getBounds());
      LINFO(" at [%d, %d, %d, %d]: %d by %d",
            nPt.i , nPt.j,
            nPt.i + prevDims.w() - 1,  nPt.j + prevDims.h() - 1,
            prevDims.w(),  prevDims.h());
    }

  // crop image and create new Visual Object
  Image< PixRGB<byte> > cImg = crop(objImg, tempR);
  objOffset.i += tempR.left();
  objOffset.j += tempR.top();
  LINFO("Overlap object: (%d,%d)", objOffset.i, objOffset.j);

  std::string wName = obj->getName() + std::string("-SAL");
  std::string wfName = wName + std::string(".png");
  obj.reset(new VisualObject(wName, wfName, cImg));

  // return match between DB and the object
  bool isSIFTfit;
  rutz::shared_ptr<VisualObjectMatch>
    cmatch2 = match(itsVisualObjectDB->getObject(mInd), obj, isSIFTfit);
  return cmatch2;
}

// ######################################################################
void Landmark::transferEvidence
( rutz::shared_ptr<Landmark> landmark2,
  bool indb2, bool intdb2, int tIndex2,
  rutz::shared_ptr<VisualObjectMatch> cmatch)
{
  rutz::shared_ptr<VisualObject> object2;
  Point2D<int> objOffset2;
  uint fnum2;

  LINFO("{%d, %d, %d}", indb2, intdb2, tIndex2);

  // insert straight to the Visual Object DB
  if(indb2)
    {
      object2    = landmark2->getObject(tIndex2);
      objOffset2 = landmark2->getOffsetCoords(tIndex2);
      fnum2      = landmark2->getVisualObjectFNum(tIndex2);

      addObject(object2, objOffset2, fnum2);
    }
  else if(intdb2)
    {
      object2    = landmark2->getTempObject(tIndex2);
      objOffset2 = landmark2->getTempOffsetCoords(tIndex2);
      fnum2      = landmark2->getTempVisualObjectFNum(tIndex2);

      addObject(object2, objOffset2, fnum2);
    }

  // cleanly delete the visual evidence from the other list
  if(indb2 || intdb2)
    landmark2->cleanDelete(indb2, intdb2, tIndex2);
}

// ######################################################################
void Landmark::cleanDelete(bool indb, bool intdb, int tIndex)
{
  if(indb)
    {
      LINFO("in db");
      if(numTempObjects() == 0 ||
         itsLatestTempVisualObjectFNum < itsVisualObjectFNum[tIndex])
        {
          LINFO("  none to move in temp");
          eraseObject(tIndex);
          eraseOffsetCoords(tIndex);
          eraseVisualObjectFNum(tIndex);
        }
      else
        {
          LINFO("  something to move in temp");
          // move the next frame on temp to the soon to be
          // un-occupied spot in the DB
          uint fnum = itsVisualObjectFNum[tIndex];
          uint ltfnum = itsLatestTempVisualObjectFNum;

          int rindex = -1;
          for(uint fn = fnum+1; fn <= ltfnum; fn++)
            {
              for(int i = int(itsTempVisualObjectFNum.size()-1); i >= 0; i--)
                {
                  uint cn = itsTempVisualObjectFNum[i];
                  if(cn == fn)
                    {
                      rindex = i;
                      LINFO("rind: %d i: %d cn: %d fn: %d", rindex, i, cn, fn);
                      i = 0; fn = ltfnum+1;
                   }
                }
            }

          if(rindex != -1)
            {
              LINFO("before: fnum: %d",itsVisualObjectFNum[tIndex]);
              itsVisualObjectDB->setObject
                (tIndex, itsTempVisualObjectDB->getObject(rindex));
              itsOffsetCoords[tIndex]     = itsTempOffsetCoords[rindex];
              itsVisualObjectFNum[tIndex] = itsTempVisualObjectFNum[rindex];
              LINFO("after : fnum: %d",itsVisualObjectFNum[tIndex]);

              LINFO("before: T: %d", numTempObjects());
              eraseTempObject(rindex);
              eraseTempOffsetCoords(rindex);
              eraseTempVisualObjectFNum(rindex);
              LINFO("after : T: %d", numTempObjects());
            }

          // update latest number for temp
          itsLatestTempVisualObjectFNum = 0;
          for(uint i = 0; i < itsTempVisualObjectFNum.size(); i++)
            {
              if(itsLatestTempVisualObjectFNum <  itsTempVisualObjectFNum[i])
                itsLatestTempVisualObjectFNum = itsTempVisualObjectFNum[i];
            }
        }

      // update latest number for vodb
      itsLatestVisualObjectFNum = 0;
      for(uint i = 0; i < itsVisualObjectFNum.size(); i++)
        {
          if(itsLatestVisualObjectFNum < itsVisualObjectFNum[i])
            itsLatestVisualObjectFNum = itsVisualObjectFNum[i];
        }
    }
  else if(intdb)
    {
      LINFO("intdb");

      eraseTempObject(tIndex);
      eraseTempOffsetCoords(tIndex);
      eraseTempVisualObjectFNum(tIndex);

      // update latest number for temp
      itsLatestTempVisualObjectFNum = 0;
      for(uint i = 0; i < itsTempVisualObjectFNum.size(); i++)
        {
          if(itsLatestTempVisualObjectFNum <  itsTempVisualObjectFNum[i])
            itsLatestTempVisualObjectFNum = itsTempVisualObjectFNum[i];
        }
    }
}

// ######################################################################
// get the position of the landmark
Point2D<int> Landmark::getPosition()
{
  uint last = numObjects() - 1;
  //uint fNum =  itsVisualObjectFNum[last];

  // get the active trackers
  std::vector<rutz::shared_ptr<Keypoint> > activeKP = getActiveKeypoints();
  LINFO("have %" ZU " active keypoints", activeKP.size());

  // get fittest tracker
  rutz::shared_ptr<KeypointTracker> fittestKPtr = getFittestKPtr();
  std::vector<rutz::shared_ptr<Keypoint> >
    fKP = fittestKPtr->getKeypoints();

  // get its last absolute loc
  Point2D<int> res = fittestKPtr->getAbsLoc();
  LINFO("%s: (%d,%d)", fittestKPtr->getName().c_str(), res. i, res.j);

  // draw the changes
  if(itsMatchWin.is_valid())
    {
      // draw the points
      int w = itsMatchWin->getDims().w()/2;
      int h = itsMatchWin->getDims().h()/2;
      Image< PixRGB<byte> > tImg(w,h,ZEROS);
      Image< PixRGB<byte> > oImg =
        itsVisualObjectDB->getObject(last)->getImage();
      for(uint i = 0; i < fKP.size(); i++)
        {
          const float x = fKP[i]->getX();
          const float y = fKP[i]->getY();
          Point2D<int> loc(int(x + 0.5F), int(y + 0.5F));
          drawDisk(oImg, loc, 2, PixRGB<byte>(255,0,0));
          LINFO("Loc: (%d,%d)",loc.i,loc.j);
        }
      inplacePaste(tImg, oImg,   Point2D<int>(0, 0));
      itsMatchWin->drawImage(tImg,w,h);
    }

  return res;
}

// ######################################################################
// get the current velocity of the landmark
// ===> USE VELOCITY LATER
Point2D<int> Landmark::getVelocity()
{
  // stationary keypoint by default
  if(numObjects() == 1)
    return Point2D<int>(0,0);

  // extrapolate speed from the last 5 points

  return Point2D<int>(0,0);
}

// ######################################################################
// get the keypoints in the current frames that are likely to be
// as determined using the trackers
std::vector<rutz::shared_ptr<Keypoint> > Landmark::getActiveKeypoints()
{
  // the number of frames the landmark has been activated
  uint last = numObjects() - 1;
  uint fNum =  itsVisualObjectFNum[last];
  uint sfNum = itsVisualObjectFNum[0];
  uint dfnum = fNum - sfNum;
  dfnum+=0;
  LINFO("fNum: %d, sfNum: %d, dfnum: %d",fNum, sfNum, dfnum);

  std::vector<rutz::shared_ptr<Keypoint> > res;
  //uint count = 0;

  // go through all the trackers
  for(uint i = 0; i < itsKPtrackers.size(); i++)
    {
      // if it is active and has length at least 3
      if((itsKPtrackers[i]->hasKeypointInFrame(fNum)) &&
         (itsKPtrackers[i]->numKeypoints() > dfnum    ||
          itsKPtrackers[i]->numKeypoints() >= MIN_LINKS))
        {
          res.push_back(itsKPtrackers[i]->getKeypointInFrame(fNum));
          //LINFO("from tracker%d: %d, %d", i,
          //      itsKPtrackers[i]->hasKeypointInFrame(fNum),
          //      itsKPtrackers[i]->numKeypoints() );
          //count++;
        }
      //if(count > 4) break;
    }

  // check it out
  if(itsMatchWin.is_valid())
    {
      // draw the points
      int w = itsMatchWin->getDims().w()/2;
      int h = itsMatchWin->getDims().h()/2;
      Image< PixRGB<byte> > tImg(w,h,ZEROS);
      Image< PixRGB<byte> >
        oImg = itsVisualObjectDB->getObject(last)->getImage();
      for(uint i = 0; i < res.size(); i++)
        {
          const float x = res[i]->getX();
          const float y = res[i]->getY();
          Point2D<int> loc(int(x + 0.5F), int(y + 0.5F));
          drawDisk(oImg, loc, 2, PixRGB<byte>(255,0,0));
          //LINFO("Loc: (%d,%d)",loc.i,loc.j);
        }
      inplacePaste(tImg, oImg,   Point2D<int>(0, 0));
      itsMatchWin->drawImage(tImg,w,h);
    }

  return res;
}

// ######################################################################
//get the longest active tracker
rutz::shared_ptr<KeypointTracker> Landmark::getFittestKPtr()
{
  // the current frame number
  uint last = numObjects() - 1;
  uint fNum =  itsVisualObjectFNum[last];

  rutz::shared_ptr<KeypointTracker> res;
  uint maxLength = 0U;

  // go through all the tracker
  for(uint i = 0; i < itsKPtrackers.size(); i++)
    {
      if((itsKPtrackers[i]->hasKeypointInFrame(fNum)) &&
         (itsKPtrackers[i]->numKeypoints() > maxLength))
        {
          res = itsKPtrackers[i];
          maxLength =  itsKPtrackers[i]->numKeypoints();
        }
    }

  LINFO("%s: %d", res->getName().c_str(), res->numKeypoints());
  return res;
}

// ######################################################################
// Prune the Visual Object keypoints of frame index temporally
void Landmark::temporalPrune(uint index)
{
  rutz::shared_ptr<VisualObject> pObj =
    itsVisualObjectDB->getObject(index);
  //std::vector< rutz::shared_ptr<Keypoint> > fkeypoints =
  //  pObj->getKeypoints();
  LDEBUG("pObj[%d] BEFORE numFeatures: %d", index, pObj->numKeypoints());

  // create a temporary list of keypoints indexes to keep
  std::vector< rutz::shared_ptr<Keypoint> > prunedfKp;

  // go through the trackers
  for(uint i = 0; i < itsKPtrackers.size(); i++)
    {
      // for the ones that has the index
      if(itsKPtrackers[i]->hasKeypointInFrame(index))
      {
        // store it
        rutz::shared_ptr<Keypoint> kp =
          itsKPtrackers[i]->getKeypointInFrame(index);
        prunedfKp.push_back(kp);
      }
    }

  // delete the keypoints that are noted as kept
  rutz::shared_ptr<VisualObject> prunedfObj
    (new VisualObject(pObj->getName(),
                      pObj->getImageFname(), pObj->getImage(),
                      pObj->getSalPoint(),
                      pObj->getFeatures(), prunedfKp));
  LDEBUG("PRUNED numFeatures: %d",prunedfObj->numKeypoints());
  itsVisualObjectDB->setObject(index, prunedfObj);
  LDEBUG("fAFTER numFeatures: %d", pObj->numKeypoints());

  // isSorted = false; DO WE NEED TO DO THIS???
}

// // ######################################################################
// // Prune the Visual Object keypoints of the last
// void Landmark::temporalPrune(rutz::shared_ptr<VisualObjectMatch> match)
// {
//   // prune the just added object
//   uint last = itsVisualObjectDB->numObjects() - 1;
//   rutz::shared_ptr<VisualObject> obj =
//    itsVisualObjectDB->getObject(last);
//   std::vector< rutz::shared_ptr<Keypoint> > keypoints =
//     obj->getKeypoints();

//   // create a temporary list of keypoints indexes to keep
//   std::vector< rutz::shared_ptr<Keypoint> > prunedKp;

//   LDEBUG("match->size: %d, kp->size: %d ",
//          match->size(), keypoints.size());
//   // go through all the keypoint matches
//   for(uint i = 0; i < match->size(); i++)
//     {
//       // go through the previous frame keypoints
//       for(uint j = 0; j < keypoints.size(); j++)
//         {
//           // if the keypoint in the match are found
//           if((*match)[i].refkp == keypoints[j])
//           {
//             // add to the temporary list
//             prunedKp.push_back(keypoints[j]);
//           }
//         }
//     }
// //       // note: the behavior of vector::erase()
//          // guarantees this code works...

//   // delete the keypoints that are noted as kept
//   LDEBUG("BEFORE numFeatures: %d",
//          itsVisualObjectDB->getObject(last)->numKeypoints());
//   rutz::shared_ptr<VisualObject> prunedObj
//     (new VisualObject(obj->getName(),
//                       obj->getImageFname(), obj->getImage(),
//                       obj->getFeatures(), prunedKp));
//   LDEBUG("PRUNED OBJ numFeatures: %d",prunedObj->numKeypoints());
//   itsVisualObjectDB->setObject(last, prunedObj);
//   LINFO("AFTER numFeatures: %d",
//         itsVisualObjectDB->getObject(last)->numKeypoints());
// }

// ######################################################################
// track the keypoints of the just added object
void Landmark::trackKeypoints(rutz::shared_ptr<VisualObjectMatch> match,
                              int mInd)
{
  uint last = itsVisualObjectDB->numObjects() - 1;
  uint fNum =  itsVisualObjectFNum[last];
  rutz::shared_ptr<VisualObject> voTst = itsVisualObjectDB->getObject(last);
  rutz::shared_ptr<VisualObject> voRef = itsVisualObjectDB->getObject(mInd);

  LINFO("we are comparing frames %d and %d",mInd, last);
  LINFO("ref->size %d, tst->size: %d match->size: %d, kp->size: %" ZU " ",
        voRef->numKeypoints(), voTst->numKeypoints(),
        match->size(), itsKPtrackers.size());

  // initialize all tst keypoints as not yet added
  std::vector<bool> voTstkpIsAdded;
  for(uint i = 0; i < voTst->numKeypoints(); i++)
    voTstkpIsAdded.push_back(false);

//   // go through all the matches
//   for(uint i = 0; i < match->size(); i++)
//     {
//       printf ("[%4d]",i);
//       // the ref keypoint
//       for(uint k = 0; k < voRef->numKeypoints(); k++)
//         if((*match)[i].refkp == voRef->getKeypoint(k)) printf("%4d ->", k);

//       // the tst keypoint
//       for(uint k = 0; k < voTst->numKeypoints(); k++)
//         if((*match)[i].tstkp == voTst->getKeypoint(k)) printf("%4d\n", k);
//    }
//   Raster::waitForKey();

  // go through all the matches
  for(uint i = 0; i < match->size(); i++)
    {
      //LINFO("[%3d]",i);

      // get the tracker with the ref keypoint
      for(uint j = 0; j < itsKPtrackers.size(); j++)
        {
           // if the keypoint is in one of the matches
          if(itsKPtrackers[j]->hasKeypointInFrame(mInd) &&
             (*match)[i].refkp == itsKPtrackers[j]->getKeypointInFrame(mInd))
             {
               // add it to the tracker
               itsKPtrackers[j]->add((*match)[i].tstkp,
                                     itsOffsetCoords[last],
                                     itsVisualObjectFNum[last]);

               // note that the tst keypoint has been added
               for(uint k = 0; k < voTst->numKeypoints(); k++)
                 {
                   if((*match)[i].tstkp == voTst->getKeypoint(k))
                     {
                       voTstkpIsAdded[k] = true;
                       //LINFO("added to %3d -> %d", k, j);
                     }
                 }
             }
        }
    }

  // create new tracker for the rest of the tst keypoints
  uint nAdded = 0U;
  for(uint i = 0; i < voTst->numKeypoints(); i++)
    {
      if(!voTstkpIsAdded[i])
        {
          rutz::shared_ptr<KeypointTracker>
            newKPtracker(new KeypointTracker
                         (sformat("KPtrk%d_%d", fNum, nAdded)));
          newKPtracker->add(voTst->getKeypoint(i),
                            itsOffsetCoords[last],
                            itsVisualObjectFNum[last]);
          itsKPtrackers.push_back(newKPtracker);
          nAdded++;
          //LINFO("Adding tracker: %d for %d", itsKPtrackers.size(), i);
        }
    }

  // prune inactive and dormant trackers from 5 frames ago
  // and that frame was in the list
  uint ndel = 0U;
  if(fNum >= FWAIT_NUM)
    {
      uint pfNum = fNum - FWAIT_NUM;

      // prune the keypoints
      std::vector<rutz::shared_ptr<KeypointTracker> >::iterator
        itr = itsKPtrackers.begin();
      while (itr < itsKPtrackers.end())
        {
          // that are inactive since pfNum and has less than MIN_LINKS chain
          if((*itr)->isInactiveSince((pfNum+1)) &&
             (*itr)->numKeypoints() < MIN_LINKS)
            {
              itr = itsKPtrackers.erase(itr); ++ ndel;
            }
          else ++ itr;
          // note: the behavior of vector::erase() guarantees this code works..
        }
      LINFO("we pruned %d inactive tracker from frame %d", ndel, pfNum);

      // pruning the keypoints off the VO
      LINFO(":{}: %d",pfNum);
      int pfNumInd = getFNumIndex(pfNum);

      if(pfNumInd != -1)
        {
          LINFO("BEFORE(%d) -> %d numFeatures: %d", pfNum, pfNumInd,
                itsVisualObjectDB->getObject(pfNumInd)->numKeypoints());
          temporalPrune(pfNumInd);
          LINFO("AFTER (%d) -> %d numFeatures: %d", pfNum, pfNumInd,
                itsVisualObjectDB->getObject(pfNumInd)->numKeypoints());
          //Raster::waitForKey();
        }
      else
        {
          LINFO("skipping %d -> %d", pfNum, pfNumInd);
          Raster::waitForKey();
        }
    }
  else
    LINFO("we pruned 0 inactive trackers");

  LINFO("we added %d  tracker for frame %d", nAdded, fNum);
  LINFO("there are %" ZU " keypoints being tracked", itsKPtrackers.size());
}

// ######################################################################
int Landmark::getFNumIndex(uint fNum)
{
  uint cInd = itsVisualObjectDB->numObjects() - 1;
  uint cfNum =  itsVisualObjectFNum[cInd];
  //LINFO("cInd: %d, (cfNum: %d, fNum: %d)",cInd,cfNum,fNum);

  // look for the index only when fNum is still bigger than current
  while (fNum <= cfNum && cInd >= 0)
    {
      //LINFO(" cInd: %d, (cfNum: %d, fNum: %d)",cInd,cfNum,fNum);
      if(fNum == cfNum) return cInd;
      if(cInd == 0) return -1;
      cInd --;
      cfNum =  itsVisualObjectFNum[cInd];
    }

    return -1;
}

// ######################################################################
Rectangle Landmark::getObjectRect(uint index)
{
  rutz::shared_ptr<VisualObject>
    vo = itsVisualObjectDB->getObject(index);
  Dims voDims = vo->getImage().getDims();
  return Rectangle(itsOffsetCoords[index], voDims);
}

// ######################################################################
rutz::shared_ptr<VisualObjectMatch>
Landmark::match(rutz::shared_ptr<VisualObject> a,
                rutz::shared_ptr<VisualObject> b, bool &isFit,
                float maxPixDist, float minfsim,
                float minscore, uint minmatch,
                float maxRotate, float maxScale, float maxShear,
                bool showMatch)
{
  // compute the matching keypoints:
  Timer tim(1000000); tim.reset();
  rutz::shared_ptr<VisualObjectMatch>
    matchRes(new VisualObjectMatch(a, b, *itsVoma));
  uint64 t = tim.get();

  // let's prune the matches:
  uint orgSize = matchRes->size();
  tim.reset();
  uint np = matchRes->prune();
  uint t2 = tim.get();

  LDEBUG("Found %u matches for (%s & %s) in %.3fms: pruned %u in %.3fms",
         orgSize, a->getName().c_str(),
         b->getName().c_str(), float(t) * 0.001F,
         np, float(t2) * 0.001F);

  // matching scores
  float kpAvgDist    = matchRes->getKeypointAvgDist();
  float afAvgDist    = matchRes->getAffineAvgDist();
  float score        = matchRes->getScore();
  bool  isSIFTaffine = matchRes->
    checkSIFTaffine(maxRotate, maxScale, maxShear);
  LDEBUG("maxPixDist: %f minfsim: %f minscore: %f minmatch: %d "
         "maxRotate: %f maxScale: %f maxShear: %f",
         maxPixDist, minfsim, minscore, minmatch,
         maxRotate, maxScale, maxShear);
  LDEBUG("kpAvgDist = %.4f|affAvgDist = %.4f|score: %.4f|aff? %d",
         kpAvgDist, afAvgDist, score, isSIFTaffine);

  if (!isSIFTaffine) LDEBUG("### bad Affine -- BOGUS MATCH");
  else if(matchRes->size() >= minscore)
    {
      // show our final affine transform:
      SIFTaffine s = matchRes->getSIFTaffine();
      LDEBUG("[tstX]   [ %- .3f %- .3f ] [refX]   [%- .3f]", s.m1, s.m2, s.tx);
      LDEBUG("[tstY] = [ %- .3f %- .3f ] [refY] + [%- .3f]", s.m3, s.m4, s.ty);
    }

  // check SIFT match
  bool isSIFTfit = (isSIFTaffine && (score >= minscore) &&
                    (matchRes->size() >= minmatch));
  LDEBUG("isSIFTfit: %d: (%d && (%f >= %f) && (%d >= %d)) ", isSIFTfit,
         isSIFTaffine, score, minscore, matchRes->size(), minmatch);

  // check sal match
  // we skip this step if the salient point is not provided
  bool isSalfit = false;
  bool salAvailable =
    (a->getSalPoint().isValid() && a->getSalPoint().isValid());
  if(isSIFTfit && salAvailable)
    {
      float sdiff = matchRes->getSalDiff();
      float sdist = matchRes->getSalDist();

      // has to pass distance proximity test (maxPixDist pixels),
      // and minfsim feature similarity
      if(sdist <= maxPixDist && sdiff >= minfsim) isSalfit = true;
      LDEBUG("isSalFit: %d: (%f <= %f && %f >= %f", isSalfit,
             sdist, maxPixDist, sdiff, minfsim);
    }
  else isSalfit = true;

  isFit = isSIFTfit && isSalfit;
  LDEBUG("isFit? %d", isFit);

  // if there are images to be displayed
  if(showMatch && itsMatchWin.is_valid() &&
     a->getImage().initialized() && b->getImage().initialized())
    {
      int w = itsMatchWin->getDims().w()/2;
      int h = itsMatchWin->getDims().h()/2;

      // get an image showing the matches and the fused image
      Image< PixRGB<byte> > mImg = matchRes->getMatchImage(1.0F);
      Image< PixRGB<byte> > fImg = matchRes->getFusedImage(0.25F);
      Image< PixRGB<byte> > tImg(2*w,2*h,ZEROS);

      inplacePaste(tImg, mImg,   Point2D<int>(0, 0));
      inplacePaste(tImg, fImg,   Point2D<int>(w, 0));

      itsMatchWin->drawImage(tImg,0,0);
    }
  return matchRes;
}

// ######################################################################
void Landmark::addObject(rutz::shared_ptr<VisualObject> obj,
                         Point2D<int> objOffset, uint fNum)
{
  // add the object to the db:
  if (itsVisualObjectDB->addObject(obj))
    {
      // note the frame number and coordinates
      itsVisualObjectFNum.push_back(fNum);
      itsOffsetCoords.push_back(objOffset);
      if(itsLatestVisualObjectFNum < fNum)
        itsLatestVisualObjectFNum = fNum;

      if(obj->getSalPoint().i != -1)
        LINFO("Added SAL VisualObject '%s' as part of %s evidence.",
              obj->getName().c_str(), itsName.c_str());
      else
        LINFO("Added NON_SAL VisualObject '%s' as part of %s evidence.",
              obj->getName().c_str(), itsName.c_str());
    }
  else
    LERROR("FAILED adding VisualObject '%s' to database -- IGNORING",
           obj->getName().c_str());
}

// ######################################################################
void Landmark::tAddObject(rutz::shared_ptr<VisualObject> obj,
                          Point2D<int> objOffset, uint fNum)
{
  // add to the tempVODB instead
  if (itsTempVisualObjectDB->addObject(obj))
    {
      // note the frame number and coordinates
      itsTempVisualObjectFNum.push_back(fNum);
      itsTempOffsetCoords.push_back(objOffset);
      if(itsLatestTempVisualObjectFNum < fNum)
        itsLatestTempVisualObjectFNum = fNum;

      if(obj->getSalPoint().i != -1)
        LINFO("Added SAL VisualObject '%s' as part of %s "
              "temporary holding",
              obj->getName().c_str(), itsName.c_str());
      else
        LINFO("Added NON_SAL VisualObject '%s' as part of %s "
              "temporary holding",
              obj->getName().c_str(), itsName.c_str());
    }
  else
    LERROR("FAILED adding VisualObject '%s' to temp holding",
           obj->getName().c_str());
}

// ######################################################################
int Landmark::match
( rutz::shared_ptr<VisualObject> obj,
  rutz::shared_ptr<VisualObjectMatch> &cmatch, int start, int end,
  float maxPixDist, float minfsim, float minscore, uint minmatch,
  float maxRotate, float maxScale, float maxShear)
{
  if(start == -1) start = 0; ASSERT(start >= 0);
  if(end == -1) end = numObjects() - 1; ASSERT(end < int(numObjects()));
  ASSERT(start <= end);

  // return -1 if match not found
  int index = findDBmatch(obj, cmatch, end-start+1, true, start,
                          maxPixDist, minfsim, minscore, minmatch,
                          maxRotate, maxScale, maxShear);
  return index;
}

// ######################################################################
int Landmark::findDBmatch
(rutz::shared_ptr<VisualObject> obj,
 rutz::shared_ptr<VisualObjectMatch> &cmatch, uint nFrames,
 bool isForward, int start,
 float maxPixDist, float minfsim, float minscore, uint minmatch,
 float maxRotate, float maxScale, float maxShear)
{
  // if db is empty return -1 right away
  if(itsVisualObjectDB->numObjects() == 0)
    { LINFO("%s DB is empty", itsName.c_str()); return -1; }

  // setup the range for vo checking
  int sInd, eInd, inc; // note: end index not included
  ASSERT((start >= 0 && start < int(numObjects())) || start == -1);
  if(start == -1 &&isForward) sInd = 0;
  else if(start == -1 && !isForward) sInd = numObjects() - 1;
  else sInd = start;

  if(isForward)
    { eInd = start+nFrames; inc = 1; }
  else
    { eInd = sInd - nFrames; if(eInd < -1) eInd = -1; inc = -1; }
  LDEBUG("Range %s DB: [%d, %d} by %d", itsName.c_str(), sInd, eInd, inc);
  int i = sInd;  int matchInd = -1;
  while(i != eInd)
    {
      LDEBUG("check %s DB(%d): %s",itsName.c_str(), i,
             itsVisualObjectDB->getObject(i)->getName().c_str());

      // check SIFT and Saliency match
      bool isFit; cmatch = match(obj, itsVisualObjectDB->getObject(i), isFit,
                                 maxPixDist, minfsim, minscore, minmatch,
                                 maxRotate, maxScale, maxShear);

      // if we didn't have a match, flip the order (assymetric matching)
      if(!isFit)
        {
          cmatch = match(itsVisualObjectDB->getObject(i), obj, isFit,
                         maxPixDist, minfsim, minscore, minmatch,
                         maxRotate, maxScale, maxShear);
        }

      // found first good match: break (maybe can do better)
      if(isFit) { matchInd = i; return matchInd; }
      i += inc;
    }
  return matchInd;
}

// ######################################################################
int Landmark::findTempDBmatch(rutz::shared_ptr<VisualObject> obj,
                              rutz::shared_ptr<VisualObjectMatch> &cmatch,
                              uint nFrames,
                              float maxPixDist, float minfsim,
                              float minscore, uint minmatch)
{
  // if temp db is empty return -1 right away
  if(itsTempVisualObjectDB->numObjects() == 0)
    { LINFO("TDB of %s is empty", itsName.c_str()); return -1; }

  // find match with the last nFrames objects in the tempDB
  int mtInd = -1;  int mintIndex;
  if(itsTempVisualObjectDB->numObjects() < nFrames)
    mintIndex = 0;
  else
    mintIndex = itsTempVisualObjectDB->numObjects() - nFrames;
  LINFO("Range %s tempDB:  [%d, %d]", itsName.c_str(), mintIndex,
        itsTempVisualObjectDB->numObjects() - 1);

  for(int i = itsTempVisualObjectDB->numObjects() - 1; i >= mintIndex; i--)
    {
      LDEBUG("check %s tempDB(%d): %s",itsName.c_str(), i,
             itsTempVisualObjectDB->getObject(i)->getName().c_str());

      // check SIFT and Saliency match
      bool isFit;
      cmatch = match(obj, itsTempVisualObjectDB->getObject(i), isFit,
                     maxPixDist, minfsim, minscore, minmatch);

      // if we didn't have a match, flip the order (asymmetric matching)
      if(!isFit)
        {
          cmatch = match(itsTempVisualObjectDB->getObject(i), obj, isFit,
                         maxPixDist, minfsim, minscore, minmatch);
        }

      // found first good match: break (maybe a bad policy)
      if(isFit){ mtInd = i; return mtInd; }
    }
  return mtInd;
}

// ######################################################################
void Landmark::moveLatestTempVisualObjectToDB()
{
  int index = itsTempVisualObjectFNum.size() - 1;
  if(index == -1)
    {
      LINFO("nothing to move in temp of %s", itsName.c_str());
      return;
    }

  LINFO("moving T(%d): %s", index,
        itsTempVisualObjectDB->getObject(index)->getName().c_str());
  moveTempVisualObjectToDB(index);
}

// ######################################################################
void Landmark::moveTempVisualObjectToDB(int index)
{
  ASSERT(index >= 0 && index < int(numTempObjects()));

  // push back the temp objects to the DB
  addObject(itsTempVisualObjectDB->getObject(index),
            itsTempOffsetCoords[index],
            itsTempVisualObjectFNum[index]);

  // update the latest number for DB
  if(itsLatestVisualObjectFNum < itsTempVisualObjectFNum[index])
    itsLatestVisualObjectFNum =  itsTempVisualObjectFNum[index];

  // erase the info off the temp list
  itsTempVisualObjectDB->eraseObject(index);
  itsTempOffsetCoords.erase(itsTempOffsetCoords.begin() + index);
  itsTempVisualObjectFNum.erase(itsTempVisualObjectFNum.begin() + index);

  // update latest number for temp
  itsLatestTempVisualObjectFNum = 0;
  for(uint i = 0; i < itsTempVisualObjectFNum.size(); i++)
    {
      if(itsLatestTempVisualObjectFNum <  itsTempVisualObjectFNum[i])
        itsLatestTempVisualObjectFNum = itsTempVisualObjectFNum[i];
    }
}

// ######################################################################
uint Landmark::numMatch(rutz::shared_ptr<Landmark> lmk,
                        float maxPixDist, float minfsim,
                        float minscore, uint minmatch)
{
  uint count = 0;

  // go through all the objects in the input Landmark
  for(uint i = 0; i < lmk->numObjects(); i++)
  {
    LDEBUG("checking lmk->object(%d)", i);
    rutz::shared_ptr<VisualObject> obj = lmk->getObject(i);

    // check with the objects in the DBlist
    for(uint j = 0; j < numObjects(); j++)
      {
        LDEBUG("     with object(%d)", j);

        // check SIFT and Saliency match
        bool isFit;
        rutz::shared_ptr<VisualObjectMatch> cmatch =
          match(obj, itsVisualObjectDB->getObject(j), isFit,
                maxPixDist, minfsim, minscore, minmatch);
        //if(itsMatchWin.is_valid())
        //  itsMatchWin->setTitle(sformat("M: obj(%d)-db(%d)",i,j).c_str());

        // if necesarry reverse the order
        if(!isFit)
        {
          cmatch = match(itsVisualObjectDB->getObject(j), obj, isFit,
                         maxPixDist, minfsim, minscore, minmatch);
          //if(itsMatchWin.is_valid())
          //  itsMatchWin->setTitle(sformat("M: db(%d)&in(%d)",i,j).c_str());
        }

        // if we pass both sal and sift tests
        if(isFit)
          {
            count++;
            LDEBUG("Match: %s(%d) & %s(%d), count: %d",
                   lmk->getName().c_str(),i, itsName.c_str(), j, count);

            // end the inner loop
            j = numObjects();
          }
      }
  }

  return count;
}

// ######################################################################
void Landmark::combine(rutz::shared_ptr<Landmark> lmk1,
                       rutz::shared_ptr<Landmark> lmk2)
{
  // add all objects in lmk1
  for(uint i = 0; i <lmk1-> numObjects(); i++)
    {
      addObject(lmk1->getObject(i),
                lmk1->getOffsetCoords(i),
                lmk1->getVisualObjectFNum(i));
    }

  // add all objects in lmk2
  for(uint i = 0; i <lmk2-> numObjects(); i++)
    {
      addObject(lmk2->getObject(i),
                lmk2->getOffsetCoords(i),
                lmk2->getVisualObjectFNum(i));
    }
}

// ######################################################################
void Landmark::append(rutz::shared_ptr<Landmark> lmk)
{
  // add all objects in the input lmk
  for(uint i = 0; i <lmk->numObjects(); i++)
    {
      addObject(lmk->getObject(i),
                lmk->getOffsetCoords(i),
                lmk->getVisualObjectFNum(i));
    }
}

// ######################################################################
struct SortObj
{
  SortObj() { };

  SortObj(const rutz::shared_ptr<VisualObject> _obj,
          const Point2D<int> _objOffset,
          const uint _fNum,
          const uint _sNum) :
    obj(_obj),
    objOffset(_objOffset),
    fNum(_fNum),
    sNum(_sNum)
  {  }

  rutz::shared_ptr<VisualObject> obj;
  Point2D<int> objOffset;
  uint fNum;
  uint sNum;

  bool operator < (const SortObj& rhs)
  {
    if(sNum != rhs.sNum) return sNum < rhs.sNum;
    else return fNum < rhs.fNum;
  }

};

void Landmark::sort(std::vector<std::string> sessionNames)
{
  std::list<SortObj> tList;
  for(uint i = 0; i < numObjects(); i++)
    {
      // save the objects, offset coordinates, and frame number
      // to a temporary place
      rutz::shared_ptr<VisualObject> obj =
        itsVisualObjectDB->getObject(i);

      // get the object name
      std::string sname = obj->getName();
      LDEBUG("session name: %s", sname.c_str());
      sname = sname.substr(0, sname.find_last_of('_'));
      sname = sname.substr(0, sname.find_last_of('_'));
      sname = sname.substr(0, sname.find_last_of('_'));

      uint j = 0;
      // check it against the names on the list
      while((j < sessionNames.size()) && (sname != sessionNames[j])) j++;
      if(j == sessionNames.size())
        LFATAL("Session not in list: %s (%s)",
               sname.c_str(), obj->getName().c_str());

      Point2D<int> objOffset = itsOffsetCoords[i];
      uint fNum = itsVisualObjectFNum[i];

      LDEBUG("B[%s] [%3d,%3d] [%3d] sNum: %d", obj->getName().c_str(),
             objOffset.i, objOffset.j, fNum, j);

      tList.push_back(SortObj(obj,objOffset,fNum,j));
    }

  tList.sort();

  rutz::shared_ptr<VisualObjectDB> vodb(new VisualObjectDB());
  std::list<SortObj>::iterator itr = tList.begin();
  uint ii = 0;
  while (itr != tList.end())
    {
      vodb->addObject((*itr).obj);
      itsOffsetCoords[ii] =  (*itr).objOffset;
      itsVisualObjectFNum[ii] = (*itr).fNum;
      itr++; ii++;
    }

  itsVisualObjectDB = vodb;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
