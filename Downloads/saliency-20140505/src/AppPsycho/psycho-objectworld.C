/*!@file AppPsycho/psycho-objectworld.C Create an object world */

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
// Primary maintainer for this file: Dan Parks <danielfp@usc.edu>
// $HeadURL$
// $Id$


#include "GUI/ViewPort3D.H"
#include "GUI/SuperQuadric.H"
#include "Util/log.H"
#include "Util/WorkThreadServer.H"
#include "Util/JobWithSemaphore.H"
#include "Component/ModelManager.H"
#include "Raster/GenericFrame.H"
#include "Image/Layout.H"
#include "Image/MatrixOps.H"
#include "Image/DrawOps.H"
#include "GUI/DebugWin.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include <stdlib.h>
#include <math.h>
#include <rutz/rand.h>
#include <nub/object.h>

class CartesianPosition
{
protected:
  std::vector<float> itsPos;
public:
  CartesianPosition(const float x, const float y, const float z, const float roll, const float pitch, const float yaw)
  {
    itsPos = std::vector<float>(6);
    itsPos[0]=x; itsPos[1]=y; itsPos[2]=z;
    itsPos[3]=roll; itsPos[4]=pitch; itsPos[5]=yaw;
  }

  inline const float& operator[](const uint lev) const;
  inline float& operator[](const uint lev);
  inline CartesianPosition& operator=(const CartesianPosition& pos);

  inline size_t size() { return itsPos.size(); }

};


inline const float& CartesianPosition::operator[](const uint lev) const
{
  ASSERT(lev>=0 && lev <=6);
  return itsPos[lev];
}

inline float& CartesianPosition::operator[](const uint lev)
{
  ASSERT(lev>=0 && lev <=6);
  return itsPos[lev];
}

inline CartesianPosition& CartesianPosition::operator=(const CartesianPosition& pos)
{
  itsPos = pos.itsPos;
  return *this;
}

inline CartesianPosition operator*(const CartesianPosition& pos, const float scale)
{
  return CartesianPosition(pos[0]*scale,pos[1]*scale,pos[2]*scale,pos[3]*scale,pos[4]*scale,pos[5]*scale);
}

inline CartesianPosition& operator*=(CartesianPosition& pos, const float scale)
{
  return (pos=pos*scale);
}

inline CartesianPosition operator+(const CartesianPosition& pos, const CartesianPosition& off)
{
  return CartesianPosition(pos[0]+off[0],pos[1]+off[1],pos[2]+off[2],pos[3]+off[3],pos[4]+off[4],pos[5]+off[5]);
}

inline CartesianPosition& operator+=(CartesianPosition& pos, const CartesianPosition& off)
{
  return (pos=pos+off);
}

inline CartesianPosition operator-(const CartesianPosition& pos, const CartesianPosition& off)
{
  return CartesianPosition(pos[0]-off[0],pos[1]-off[1],pos[2]-off[2],pos[3]-off[3],pos[4]-off[4],pos[5]-off[5]);
}

inline CartesianPosition& operator-=(CartesianPosition& pos, const CartesianPosition& off)
{
  return (pos=pos-off);
}


// 3D version of rectangle
class Box
{
protected:
  // Center position
  CartesianPosition itsPos;
  // 3D dims
  CartesianPosition itsDims;
public:
  Box(CartesianPosition pos, CartesianPosition dims) :
    itsPos(pos),
    itsDims(dims)
  {
  }

  bool dimContained(CartesianPosition pnt, size_t i)
  {
    ASSERT(i>=0 && i<=itsDims.size());
    // Check lower bound
    if(pnt[i] < itsPos[i]-itsDims[i]/2)
      return false;
    // Check upper bound
    if(pnt[i] > itsPos[i]+itsDims[i]/2)
      return false;
    return true;
  }

  bool contains(CartesianPosition pnt)
  {
    for(size_t i=0;i<itsDims.size();i++)
      { 
	if(!this->dimContained(pnt,i))
	  return false;
      }
    return true;
  }
  float getVolume() { return itsDims[0]*itsDims[1]*itsDims[2]; }
  CartesianPosition getPosition() { return itsPos;}
  CartesianPosition getDimensions() { return itsDims;}
};

class Obj3DComponent : public nub::object
{
protected:
  // Type of object
  std::string itsType;
  rutz::shared_ptr<ViewPort3D> itsViewPort;
  CartesianPosition itsPos;
  PixRGB<byte> itsColor;

public:
  virtual void draw() = 0;
  CartesianPosition getPosition() { return itsPos; }
  PixRGB<byte> getColor() { return itsColor; }
  void setPosition(CartesianPosition pos) { this->itsPos = pos; }

protected:
  Obj3DComponent(std::string type,rutz::shared_ptr<ViewPort3D> vp,CartesianPosition pos,PixRGB<byte> color) :
    itsType(type),
    itsViewPort(vp),
    itsPos(pos),
    itsColor(color)
  {

  }
};

class Obj3DRectangle : public Obj3DComponent
{
  float width;
  float height;

public:
  Obj3DRectangle(rutz::shared_ptr<ViewPort3D> vp, CartesianPosition pos, PixRGB<byte> color, float width, float height) 
    : Obj3DComponent("Rectangle",vp,pos,color)
  {
    this->width = width;
    this->height = height;
  }

  virtual void draw()
  {
    Point3D<float> posPt(itsPos[0],itsPos[1],itsPos[2]);
    Point3D<float> rotPt(itsPos[3],itsPos[4],itsPos[5]);
    itsViewPort->drawRectangle(posPt,rotPt,width,height,itsColor);
  }
};
class Obj3DCircle : public Obj3DComponent
{
  float radius;
public:
  Obj3DCircle(rutz::shared_ptr<ViewPort3D> vp, CartesianPosition pos, PixRGB<byte> color, float radius) 
  : Obj3DComponent("Rectangle",vp,pos,color)
  {
    this->radius = radius;
  }

  virtual void draw()
  {
    Point3D<float> posPt(itsPos[0],itsPos[1],itsPos[2]);
    Point3D<float> rotPt(itsPos[3],itsPos[4],itsPos[5]);
    itsViewPort->drawCircle(posPt,rotPt,radius,itsColor);
  }
};

class Obj3DBox : public Obj3DComponent
{
  Point3D<float> size;
public:
  Obj3DBox(rutz::shared_ptr<ViewPort3D> vp, CartesianPosition pos, PixRGB<byte> color, Point3D<float> size) 
  : Obj3DComponent("Rectangle",vp,pos,color)
  {
    this->size = size;
  }

  virtual void draw()
  {
    Point3D<float> posPt(itsPos[0],itsPos[1],itsPos[2]);
    Point3D<float> rotPt(itsPos[3],itsPos[4],itsPos[5]);
    itsViewPort->drawBox(posPt,rotPt,size,itsColor);
  }
};

class Obj3DSphere : public Obj3DComponent
{
  Point3D<float> size;
public:
  Obj3DSphere(rutz::shared_ptr<ViewPort3D> vp, CartesianPosition pos, PixRGB<byte> color, Point3D<float> size) 
  : Obj3DComponent("Rectangle",vp,pos,color)
  {
    this->size = size;
  }

  virtual void draw()
  {
    Point3D<float> posPt(itsPos[0],itsPos[1],itsPos[2]);
    Point3D<float> rotPt(itsPos[3],itsPos[4],itsPos[5]);
    itsViewPort->drawSphere(posPt,rotPt,size,itsColor);
  }
};

class Obj3DCylinder : public Obj3DComponent
{
  float radius;
  float length;
public:
  Obj3DCylinder(rutz::shared_ptr<ViewPort3D> vp, CartesianPosition pos, PixRGB<byte> color, float radius, float length) 
  : Obj3DComponent("Rectangle",vp,pos,color)
  {
    this->radius = radius;
    this->length = length;
  }

  virtual void draw()
  {
    Point3D<float> posPt(itsPos[0],itsPos[1],itsPos[2]);
    Point3D<float> rotPt(itsPos[3],itsPos[4],itsPos[5]);
    itsViewPort->drawCylinder(posPt,rotPt,radius,length,itsColor);
  }
};

class Obj3DComposite : public Obj3DComponent
{
  // Note order of members is important, as components are drawn in this order
  std::vector<rutz::shared_ptr<Obj3DComponent> > members;
  std::vector< CartesianPosition > offsets;
public:
  Obj3DComposite(rutz::shared_ptr<ViewPort3D> vp, CartesianPosition pos, PixRGB<byte> color, 
		 std::vector<rutz::shared_ptr<Obj3DComponent> > members, std::vector< CartesianPosition > offsets)
  : Obj3DComponent("Rectangle",vp,pos,color)
  {
    this->members = members;
    this->offsets = offsets;
    ASSERT(offsets.size() == members.size());
  }

  virtual void draw()
  {
    for(size_t i=0;i<members.size();i++)
      {
	// Calculate transform between offset and global coordinate frame
	members[i]->setPosition(itsPos+offsets[i]);
	members[i]->draw();
      }
  }


};

// Subclasses should modify update function
// Simple constant velocity trajectory generator, with bounds
class ObjTrajectory
{
protected:
  Box itsBounds;
  CartesianPosition itsCurPos;
  CartesianPosition itsCurVel;
  //! Integration window 
  float itsIntegStep;
  //! Change in time per call (for constant velocity, iterStep==timeStep is ok, but with acceleration, should be integStep << timeStep)
  float itsTimeStep;
public:
  ObjTrajectory(Box bounds, CartesianPosition initPos, CartesianPosition initVel, float integStep=0.01, float timeStep=0.1) :
    itsBounds(bounds),
    itsCurPos(initPos),
    itsCurVel(initVel)
  {
    itsIntegStep = integStep;
    itsTimeStep = timeStep;
    // Integration must be within or equal to the timestep
    ASSERT(itsIntegStep<=itsTimeStep);
    // The bounds must contain the initial position of the trajectory
    ASSERT(itsBounds.contains(itsCurPos));
  }
  
  void update()
  {
    for(int iter=0;iter*itsIntegStep<itsTimeStep;iter++)
      {
	CartesianPosition tmp = itsCurPos + itsCurVel*itsIntegStep;
	if(!itsBounds.contains(tmp))
	  {
	for(size_t i=0;i<tmp.size();i++)
	  {
	    // Change velocity direction, based on any bound has been exceeded
	    if(!itsBounds.dimContained(tmp,i))
	      {
		itsCurVel[i]=-itsCurVel[i];
	      }
	  }
	  }
	else
	  itsCurPos = tmp;
      }
  }

  CartesianPosition getPosition(){ return itsCurPos; }

};


rutz::shared_ptr<Obj3DComponent> initObject1(rutz::shared_ptr<ViewPort3D> vp)
{
  CartesianPosition z(0,0,0,0,0,0);
  float basew=70,baseh=100;
  Obj3DRectangle *rbase = new Obj3DRectangle(vp,z,PixRGB<byte>(256,0,0),basew,baseh);
  float r1w=20,r1h=20;
  Obj3DRectangle *r1 = new Obj3DRectangle(vp,z,PixRGB<byte>(0,256,0),r1w,r1h);
  float c1r=8;
  Obj3DCircle *c1 = new Obj3DCircle(vp,z,PixRGB<byte>(0,0,256),c1r);
  Obj3DCircle *c2 = new Obj3DCircle(vp,z,PixRGB<byte>(0,0,256),c1r);
  std::vector<rutz::shared_ptr<Obj3DComponent> > members;
  members.push_back(rutz::shared_ptr<Obj3DComponent>(rbase));
  members.push_back(rutz::shared_ptr<Obj3DComponent>(r1));
  members.push_back(rutz::shared_ptr<Obj3DComponent>(c1));
  members.push_back(rutz::shared_ptr<Obj3DComponent>(c2));
  std::vector<CartesianPosition > offsets;
  offsets.push_back(CartesianPosition(0,0,-1,0,0,0));
  offsets.push_back(CartesianPosition(0,-30,0,0,0,0));
  offsets.push_back(CartesianPosition(20,30,0,0,0,0));
  offsets.push_back(CartesianPosition(-20,30,0,0,0,0));
  Obj3DComposite *comp = new Obj3DComposite(vp,z,PixRGB<byte>(0,0,0),members,offsets);
  return rutz::shared_ptr<Obj3DComponent>(comp);
}

int main(int argc, char *argv[]){

  ModelManager manager("Test Object World");

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);
  // let's get all our ModelComponent instances started:
  manager.start();

  //Normal operation

  rutz::shared_ptr<ViewPort3D> vp = rutz::shared_ptr<ViewPort3D>(new ViewPort3D(320,240, false, true, false));
  //ViewPort3D vp(320,240);
  vp->setCamera(Point3D<float>(0,0,350), Point3D<float>(0,0,0));

  CartesianPosition trajVel(2,2,0,0,0,0);
  rutz::shared_ptr<Obj3DComponent> obj1 = initObject1(vp);
  CartesianPosition boundSize(100,100,4,0,0,0);
  CartesianPosition z(0,0,0,0,0,0);
  Box bounds = Box(z,boundSize);
  ObjTrajectory traj1 = ObjTrajectory(bounds,z,trajVel);
  int rot = 0;
  while(1)
  {
    vp->initFrame();
    
    rot = ((rot +1)%360);

    std::vector<Point2D<float> > contour;

    Point3D<float> ctr = Point3D<float>(60,60,0);
    
    traj1.update();
    obj1->setPosition(traj1.getPosition());
    obj1->draw();


    Image<PixRGB<byte> > img = flipVertic(vp->getFrame());
    ofs->writeRGB(img, "ViewPort3D", FrameInfo("ViewPort3D", SRC_POS));
    usleep(10000);
  }


  exit(0);

}


