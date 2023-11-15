/*!@file SceneUnderstanding/Objects.C  */

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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/Objects.C $
// $Id: Objects.C 13765 2010-08-06 18:56:17Z lior $
//

#ifndef Objects_C_DEFINED
#define Objects_C_DEFINED

#include "Image/DrawOps.H"
#include "Image/MathOps.H"
//#include "Image/OpenCVUtil.H"
#include "Image/Kernels.H"
#include "Image/FilterOps.H"
#include "Image/Transforms.H"
#include "Image/fancynorm.H"
#include "Image/Convolutions.H"
#include "Image/MatrixOps.H"
#include "Simulation/SimEventQueue.H"
#include "plugins/SceneUnderstanding/Objects.H"
#include "GUI/DebugWin.H"
#include <math.h>
#include <fcntl.h>
#include <limits>
#include <string>

const ModelOptionCateg MOC_Objects = {
  MOC_SORTPRI_3,   "Objects-Related Options" };

// Used by: SimulationViewerEyeMvt
const ModelOptionDef OPT_ObjectsShowDebug =
  { MODOPT_ARG(bool), "ObjectsShowDebug", &MOC_Objects, OPTEXP_CORE,
    "Show debug img",
    "Objects-debug", '\0', "<true|false>", "false" };


// ######################################################################
Objects::Objects(OptionManager& mgr, const std::string& descrName,
    const std::string& tagName) :
  SimModule(mgr, descrName, tagName),
  SIMCALLBACK_INIT(SimEventGeons3DOutput),
  SIMCALLBACK_INIT(SimEventSaveOutput),
  SIMCALLBACK_INIT(SimEventUserInput),
  itsShowDebug(&OPT_ObjectsShowDebug, this),
  itsProb(0)
{

  initRandomNumbers();

  Table* table = new Table();
  table->pos = Point3D<double>(-3.60,-7.30,-5.00);
  table->rotation = Point3D<double>(0.31,-0.15,0.31);
  table->tableWidth = 6.01;
  table->tableLength = 4.50;
  table->legHeight = 4.00;
  table->legWidth = 0.50;

  //itsObjectsState.push_back(ObjectState(TABLE, Point3D<float>(0,0,0), Point3D<float>(0,0,0)));
  itsObjectsState.push_back(table);

  //Table* table2 = new Table();
  //table2->pos = Point3D<double>(-2.60,-7.30,-5.00);
  //table2->rotation = Point3D<double>(0.31,-0.15,0.31);
  //table2->tableWidth = 6.01;
  //table2->tableLength = 4.50;
  //table2->legHeight = 4.00;
  //table2->legWidth = 0.50;
  //itsObjectsState.push_back(table2);


}

// ######################################################################
Objects::~Objects()
{
}

// ######################################################################
void Objects::onSimEventGeons3DOutput(SimEventQueue& q,
                                  rutz::shared_ptr<SimEventGeons3DOutput>& e)
{
  itsGeons = e->getGeons();
  itsProb =  e->getProb();
  evolve(q);

  //q.post(rutz::make_shared(new SimEventObjectsOutput(this, itsEdgesState, itsCornersState)));

}

// ######################################################################
void Objects::onSimEventSaveOutput(SimEventQueue& q, rutz::shared_ptr<SimEventSaveOutput>& e)
{
  if (itsShowDebug.getVal())
    {
      // get the OFS to save to, assuming sinfo is of type
      // SimModuleSaveInfo (will throw a fatal exception otherwise):
      nub::ref<FrameOstream> ofs =
        dynamic_cast<const SimModuleSaveInfo&>(e->sinfo()).ofs;
      Layout<PixRGB<byte> > disp = getDebugImage(q);
      ofs->writeRgbLayout(disp, "Objects", FrameInfo("Objects", SRC_POS));
    }
}

// ######################################################################
void Objects::onSimEventUserInput(SimEventQueue& q, rutz::shared_ptr<SimEventUserInput>& e)
{

  LINFO("Got event %s %ix%i key=%i",
      e->getWinName(),
      e->getMouseClick().i,
      e->getMouseClick().j,
      e->getKey());

  if (e->getMouseClick().isValid())
  {
  }

  ObjectState* object = itsObjectsState[0];
  Table* table = (Table*)object;
  switch(e->getKey())
  {
    case 98: //up
      object->pos.x -= 0.1;
      break;
    case 104: //down
      object->pos.x += 0.1;
      break;
    case 100: //left
      object->pos.y -= 0.1;
      break;
    case 102: //right
      object->pos.y += 0.1;
      break;
    case 21: //=
      object->pos.z += 0.1;
      break;
    case 20: //-
      object->pos.z -= 0.1;
      break;
    case 38: //a
      object->rotation.x += 1*(M_PI/180);
      break;
    case 52: //z
      object->rotation.x -= 1*(M_PI/180);
      break;
    case 39: //s
      object->rotation.y += 1*(M_PI/180);
      break;
    case 53: //x
      object->rotation.y -= 1*(M_PI/180);
      break;
    case 40: //d
      object->rotation.z += 1*(M_PI/180);
      break;
    case 54: //c
      object->rotation.z -= 1*(M_PI/180);
      break;

    case 10: //1
      table->legHeight -= 0.1;
      break;
    case 24: //q
      table->legHeight += 0.1;
      break;
    case 11: //2
      table->tableWidth += 0.01;
      break;
    case 25: //w
      table->tableWidth -= 0.01;
      break;
    case 12: //3
      table->tableLength += 0.01;
      break;
    case 26: //e
      table->tableLength -= 0.01;
      break;
  }

  LINFO("Pos(%0.2f,%0.2f,%0.2f), rotation((%0.2f,%0.2f,%0.2f), size(%0.2f,%0.2f,%0.2f)",
      object->pos.x, object->pos.y, object->pos.z,
      object->rotation.x, object->rotation.y, object->rotation.z,
      table->tableWidth, table->tableLength,table->legHeight);


  evolve(q);

}

void Objects::calcObjectLikelihood(ObjectState& object)
{

}


std::vector<Geons3D::GeonState> Objects::getPrior(ObjectState* object)
{

  std::vector<Geons3D::GeonState> geons;

  switch(object->type)
  {
    case TABLE:
      Table* table = (Table*)object;

      float legXOffset = table->tableWidth/2 - table->legWidth/2;
      float legYOffset = table->tableLength/2 - table->legWidth/2;

      geons.push_back(
          Geons3D::GeonState(Geons3D::BOX,
            Point3D<double>(0, 0, 0+table->legHeight/2),
            //Point3D<double>(table->pos.x, table->pos.y, table->pos.z+table->legHeight/2),
            Point3D<double>(0,0,0),
            //table->rotation,
            Point3D<double>(1, 1, 1),
            Point3D<double>(table->tableWidth,table->tableLength,0.40) ));
      geons.push_back(
          Geons3D::GeonState(Geons3D::BOX,
            Point3D<double>(0-legXOffset, 0-legYOffset,0),
            //Point3D<double>(table->pos.x-legXOffset, table->pos.y-legYOffset,table->pos.z),
            Point3D<double>(0,0,0),
            Point3D<double>(1, 1, 1),
            Point3D<double>(table->legWidth,table->legWidth,table->legHeight) ));
      geons.push_back(
          Geons3D::GeonState(Geons3D::BOX,
            Point3D<double>(0+legXOffset, 0-legYOffset,0),
            //Point3D<double>(table->pos.x+legXOffset, table->pos.y-legYOffset,table->pos.z),
            Point3D<double>(0,0,0),
            Point3D<double>(1, 1, 1),
            Point3D<double>(table->legWidth,table->legWidth,table->legHeight) ));
      geons.push_back(
          Geons3D::GeonState(Geons3D::BOX,
            Point3D<double>(0-legXOffset, 0+legYOffset,0),
            //Point3D<double>(table->pos.x-legXOffset, table->pos.y+legYOffset,table->pos.z),
            Point3D<double>(0,0,0),
            Point3D<double>(1, 1, 1),
            Point3D<double>(table->legWidth,table->legWidth,table->legHeight) ));
      geons.push_back(
          Geons3D::GeonState(Geons3D::BOX,
            Point3D<double>(0+legXOffset, 0+legYOffset,0),
            //Point3D<double>(table->pos.x+legXOffset, table->pos.y+legYOffset,table->pos.z),
            Point3D<double>(0,0,0),
            Point3D<double>(1, 1, 1),
            Point3D<double>(table->legWidth,table->legWidth,table->legHeight) ));
      break;
  }

  return geons;

}

Objects::ObjectState* Objects::proposeState(ObjectState* object)
{
// ######################################################################

  object->lastSample = (object->lastSample +1)%2;

  //TODO: Check for object type
  //Table* table = (Table*)object;

  Table* table = new Table();
  table->pos = Point3D<double>(-3.60,-7.30,-5.00);
  table->rotation = Point3D<double>(0.31,-0.15,0.31);
  table->tableWidth = 6.01;
  table->tableLength = 4.50;
  table->legHeight = 4.00;
  table->legWidth = 0.50;

  switch(object->lastSample)
  {
    case 0:
      {
        Point3D<double> randLoc(
            randomDoubleFromNormal(0.10),
            randomDoubleFromNormal(0.10),
            randomDoubleFromNormal(0.10));
        table->pos += randLoc;
      }
      break;
    case 1:
      {
        Point3D<double> randRot(
            randomDoubleFromNormal(0.5)*M_PI/180,
            randomDoubleFromNormal(0.5)*M_PI/180,
            randomDoubleFromNormal(0.5)*M_PI/180);
        table->rotation += randRot;
      }
      break;
    case 2:
      //newSketch.params.x += randomDoubleFromNormal(0.25);
      //if (newSketch.params.x <= 0.1)
      //  newSketch.params.x = 0.1;
      break;
    case 3:
      //newSketch.params.y += randomDoubleFromNormal(0.25);
      //if (newSketch.params.y <= 0.1)
      //  newSketch.params.y = 0.1;
      break;
  }

  return table;

}


// ######################################################################
void Objects::evolve(SimEventQueue& q)
{
  //double totalProb = 0;

  for(uint i=0; i<itsObjectsState.size(); i++)
  {
    ObjectState* newObject = proposeState(itsObjectsState[i]);

    //ObjectState newObject = itsObjectsState[i];
    //calcObjectLikelihood(itsObjectsState[i]);



    //send prior;
    std::vector<Geons3D::GeonState> geons = getPrior(newObject);
    q.post(rutz::make_shared(new SimEventGeons3DPrior(this, geons,
            newObject->pos, newObject->rotation)));

    delete newObject;
  }
}



Layout<PixRGB<byte> > Objects::getDebugImage(SimEventQueue& q)
{
  //evolve(q);
  Layout<PixRGB<byte> > outDisp;

  //Image<float> input(512,512, ZEROS);
  //for(uint i=0; i<itsV1Edges.size(); i++)
  //    input.setVal(itsV1Edges[i].pos, itsV1Edges[i].prob);
  //inplaceNormalize(input, 0.0F, 255.0F);

  //for(uint i=0; i<itsLinesState.size(); i++)
  //{
  //  LineState line = itsLinesState[i];
  //  //drawLine(perc, line.pos, line.ori, line.length, PixRGB<byte>((float)line.nProb,0,0), 1);
  // // if (line.prob > -500)
  //  {
  //    drawLine(perc, line.pos, line.ori, line.length, PixRGB<byte>(255,0,0), 1);
  //  //  LINFO("%i %ix%i %f %f %f",
  //  //      i, line.pos.i, line.pos.j, line.ori, line.length, line.prob);
  //  }
  //}


  //outDisp = hcat(toRGB(Image<byte>(input)), perc);

  return outDisp;

}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif

