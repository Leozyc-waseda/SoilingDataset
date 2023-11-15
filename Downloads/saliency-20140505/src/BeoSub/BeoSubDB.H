/*!@file BeoSub/BeoSubDB.H */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2003   //
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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubDB.H $
// $Id: BeoSubDB.H 6003 2005-11-29 17:22:45Z rjpeters $
//

#ifndef BEOSUBDB_H_DEFINED
#define BEOSUBDB_H_DEFINED

#include "Util/Angle.H"
#include "Image/Image.H"
#include "Image/Pixels.H"

#include "BeoSub/Attitude.H"

#include <fstream>
#include <cstdlib>
#include <map>
#include <list>

struct MappingData{
  std::string itsImgFilename;
  Angle itsHeading, itsPitch, itsRoll;
  float itsXpos, itsYpos, itsZpos;
  float itsFballast, itsRballast, itsRthruster, itsLthruster;
};

typedef std::pair <std::string, MappingData> MapPair;

//! A BeoSub mapping database
/*! Structure is as follows:
  IMAGE, COMPASS(heading pitch roll), ACTUATORS (ballast[front rear], thrusters[left right]), POSITION (x, y, z)

  NOTE: zPos is based off of the depth sensor
*/
class BeoSubDB
{
public:
  // ######################################################################
  /*! @name Constructors and Destructors */
  //@{

  //! Default Constructor
  /*! This constructor reads in from a hardcoded directory all database data and reads it into a map*/
  inline BeoSubDB();

  //! Destructor
  inline ~BeoSubDB();

  inline bool initializeDatabase(std::list<std::string> filenames, std::string destFilename);

  inline bool loadDatabase(std::string filename);

  inline void saveDatabase(std::map<std::string, MappingData> inDB, std::string filename);

  inline MappingData getMappingData(std::string name);

  inline uint getNumEntries();

  inline void getDirections(MappingData from, MappingData to, Attitude& directions, float& dist);

  //@}

private:
  // ######################################################################
  MappingData inData;       //!< structure used to read in and store database members
  std::map<std::string, MappingData> itsDatabase;
  std::ifstream inFile;
  std::ofstream outFile;
};


// ######################################################################
// ########## Inlined methods:
// ######################################################################
inline BeoSubDB::BeoSubDB()
{
}

// ######################################################################
inline BeoSubDB::~BeoSubDB()
{ }

// ######################################################################
//! Returns true if files found, false otherwise
inline bool BeoSubDB::initializeDatabase(std::list<std::string> filenames, std::string destFilename){
  std::string tempImg;
  float tempHeading = 0.0, tempPitch = 0.0, tempRoll = 0.0, tempXpos = 0.0, tempYpos = 0.0, tempZpos = 0.0, tempFballast = 0.0, tempRballast = 0.0, tempLthruster = 0.0, tempRthruster = 0.0;
  std::list<std::string>::iterator fileIter;

  for(fileIter = filenames.begin(); fileIter != filenames.end(); fileIter++){
    inFile.open((*fileIter).c_str(), std::ifstream::in);
    if(inFile.fail()){
      LINFO("File %s is not a valid filename. Please specify a valid filename.", (*fileIter).c_str());
      return false;
    }

    //read in data entry from file
    inFile>>tempImg;
    inFile>>tempHeading;
    inFile>>tempPitch;
    inFile>>tempRoll;
    inFile>>tempZpos;
    inFile>>tempFballast;
    inFile>>tempRballast;
    inFile>>tempLthruster;
    inFile>>tempRthruster;
    inFile>>tempXpos;
    inFile>>tempYpos;
    // printf("Loading-- %s: Heading %f, Pitch %f, Roll %f, Depth %f, \nFront Ballast %f, Rear Ballast %f, Left Thruster %f, Right Thruster %f, \nX Position %f, Y Position %f\n\n", tempImg, tempHeading, tempPitch, tempRoll, tempZpos, tempFballast, tempRballast, tempLthruster, tempRthruster, tempXpos, tempYpos);
    //convert to appropriate datatypes
    inData.itsImgFilename = tempImg;
    inData.itsHeading.setVal(tempHeading);
    inData.itsPitch.setVal(tempPitch);
    inData.itsRoll.setVal(tempRoll);
    inData.itsFballast = tempFballast;
    inData.itsRballast = tempRballast;
    inData.itsLthruster = tempLthruster;
    inData.itsRthruster = tempRthruster;
    inData.itsXpos = tempXpos;
    inData.itsYpos = tempYpos;
    inData.itsZpos = tempZpos;
    //printf("Saving-- %s: Heading %f, Pitch %f, Roll %f, Depth %f, \nFront Ballast %f, Rear Ballast %f, Left Thruster %f, Right Thruster %f, \nX Position %f, Y Position %f\n\n", inData.itsImgFilename, inData.itsHeading.getVal(), inData.itsPitch.getVal(), inData.itsRoll.getVal(), inData.itsZpos, inData.itsFballast, inData.itsRballast, inData.itsLthruster, inData.itsRthruster, inData.itsXpos, inData.itsYpos);
    //read entry into database map
    itsDatabase.insert(MapPair(inData.itsImgFilename, inData));

    inFile.close();
  }

  saveDatabase(itsDatabase, destFilename);
  return true;
}

// ######################################################################
//! Returns true if file found, false otherwise
inline bool BeoSubDB::loadDatabase(std::string filename){
  std::string tempImg;
  float tempHeading = 0.0, tempPitch = 0.0, tempRoll = 0.0, tempXpos = 0.0, tempYpos = 0.0, tempZpos = 0.0, tempFballast = 0.0, tempRballast = 0.0, tempLthruster = 0.0, tempRthruster = 0.0;

  inFile.open(filename.c_str(), std::ifstream::in);
  if(inFile.fail()){
    LINFO("Database read failed. Please specify a valid filename.");
    return false;
  }

  while(!inFile.eof()){//until all data entries are read...
    //read in data entry from file
    inFile>>tempImg;
    inFile>>tempHeading;
    inFile>>tempPitch;
    inFile>>tempRoll;
    inFile>>tempZpos;
    inFile>>tempFballast;
    inFile>>tempRballast;
    inFile>>tempLthruster;
    inFile>>tempRthruster;
    inFile>>tempXpos;
    inFile>>tempYpos;

    //convert to appropriate datatypes
    inData.itsImgFilename = tempImg;
    inData.itsHeading = tempHeading;
    inData.itsPitch = tempPitch;
    inData.itsRoll = tempRoll;
    inData.itsFballast = tempFballast;
    inData.itsRballast = tempRballast;
    inData.itsLthruster = tempLthruster;
    inData.itsRthruster = tempRthruster;
    inData.itsXpos = tempXpos;
    inData.itsYpos = tempYpos;
    inData.itsZpos = tempZpos;
    //NOTE: may need to take care of line ends here. FIX?
    //read entry into databasemap
    itsDatabase.insert(MapPair(inData.itsImgFilename, inData));
  }
  inFile.close();
  return true;
}

// ######################################################################
//! Saves map to file
inline void BeoSubDB::saveDatabase(std::map<std::string, MappingData> inDB, std::string filename){
  itsDatabase = inDB;
  std::map<std::string, MappingData>::iterator dbIter = inDB.begin();
  outFile.open(filename.c_str(), std::ifstream::out);

  if(outFile.fail()){
    LINFO("Database file %s not found. Please specify a valid filename", filename.c_str());
    return;
  }
  //printf("Database size is %d elements\n", inDB.size());

  while(dbIter != inDB.end()){
    outFile<<((*dbIter).second).itsImgFilename<<"\t";
    outFile<<((*dbIter).second).itsHeading.getVal()<<"\t";
    outFile<<((*dbIter).second).itsPitch.getVal()<<"\t";
    outFile<<((*dbIter).second).itsRoll.getVal()<<"\t";
    outFile<<((*dbIter).second).itsZpos<<"\t";
    outFile<<((*dbIter).second).itsFballast<<"\t";
    outFile<<((*dbIter).second).itsRballast<<"\t";
    outFile<<((*dbIter).second).itsLthruster<<"\t";
    outFile<<((*dbIter).second).itsRthruster<<"\t";
    outFile<<((*dbIter).second).itsXpos<<"\t";
    outFile<<((*dbIter).second).itsYpos;
    outFile<<"\n";

    dbIter++;
  }
  outFile.close();
  return;
}

// ######################################################################
//! Returns map element at key "name"
inline MappingData BeoSubDB::getMappingData(std::string name){
  //NOTE: should have a check if the element exists in the database. However, map doesnt have a simple check for this that I see. FIX?
  return itsDatabase[name];
}

// ######################################################################
//! Returns size of the database
inline uint BeoSubDB::getNumEntries(){
  return itsDatabase.size();
}

// ######################################################################
//! This function returns by reference the attitude and distance necessary to reach a target state given the current state (represented in MappingData objects)
inline void BeoSubDB::getDirections(MappingData from, MappingData to, Attitude& directions, float& dist){
  float deltaX = to.itsXpos - from.itsXpos;
  float deltaY = to.itsYpos - from.itsYpos;

  dist = sqrt((deltaX*deltaX) + (deltaY*deltaY)); //needed distance
  float angle = (atan2(deltaY, deltaX) * (180/3.14159265)); //needed heading in degrees;

  directions.heading = angle;
  directions.pitch = to.itsPitch;
  directions.roll = to.itsRoll;
  directions.depth = to.itsZpos;
  //NOTE: what to do about time vars? nothing?
}
#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
