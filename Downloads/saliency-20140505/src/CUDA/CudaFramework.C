/*!@file CUDA/CudaFramework.C  Framework for specifying layout of images  */

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
// Primary maintainer for this file:
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaFramework.C $
// $Id: CudaFramework.C 12962 2010-03-06 02:13:53Z irock $
//

#include "CudaFramework.H"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <errno.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include "CUDA/CudaImage.H"
#include "CUDA/CudaImageSet.H"
#include "Image/Pixels.H"
#include "Image/MathOps.H"
#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include "CudaImageDisplay.h"
#include <unistd.h>
#include "Raster/Raster.H"
#include "Raster/PngWriter.H"
#include "CUDA/CudaCutPaste.H"
#include "Component/ModelManager.H"
#include "CUDA/CudaSaliency.H"
#include "Image/SimpleFont.H"
#include "Image/DrawOps.H"
#include "Raster/Raster.H"
#include "Raster/GenericFrame.H"
#include "Media/FrameSeries.H"
#include "CUDA/CudaDrawOps.H"
#include "CUDA/CudaShapeOps.H"


// ######################################################################
CudaFramework::CudaFramework()
{
  canvasModified = true;
}


// ######################################################################
CudaFramework::~CudaFramework()
{
}


// ######################################################################
int CudaFramework::getW()
{
  return w;
}


// ######################################################################
int CudaFramework::getH()
{
  return h;
}


// ######################################################################
int CudaFramework::getCanvasW()
{
  return canvas_image.getWidth();
}


// ######################################################################
int CudaFramework::getCanvasH()
{
  return canvas_image.getHeight();
}


// ######################################################################
CudaImage<PixRGB<float> >& CudaFramework::getCanvas()
{
  return canvas_image;
}


// ######################################################################
Point2D<int> CudaFramework::getPoint(int index)
{
  return point_list[index];
}

// ######################################################################
int CudaFramework::getDev()
{
  return dev1;
}


// ######################################################################
MemoryPolicy CudaFramework::getMP()
{
  return mp1;
}


// ######################################################################
int CudaFramework::getLockAtt()
{
return  pthread_mutex_trylock(&canvas_lock);
}


// ######################################################################
bool CudaFramework::getCanvasModified()
{
  return canvasModified;
}


// ######################################################################
void CudaFramework::setMutexDestroy()
{
 pthread_mutex_destroy(&canvas_lock);
}


// ######################################################################
void CudaFramework::setMutexUnlock()
{
 pthread_mutex_unlock(&canvas_lock);
}


// ######################################################################
void CudaFramework::setCanvasModified(bool new_state)
{
  canvasModified = new_state;
}


// ######################################################################
void CudaFramework::setPoint(int index,Point2D<int> pt)
{
  point_list[index] = pt;
}


// ######################################################################
void CudaFramework::setDev(int dev)
{
  dev1 = dev;
}


// ######################################################################
void CudaFramework::setMP(MemoryPolicy mp)
{
  mp1 = mp;
}


// ######################################################################
void CudaFramework::setW(int W)
{
  w = W;
}


// ######################################################################
void CudaFramework::setH(int H)
{
  h = H;
}


// ######################################################################
void CudaFramework::updateCanvas(int index, CudaImage<PixRGB<float> > &src)
{
  pthread_mutex_lock(&canvas_lock);
  cudaInplacePaste(canvas_image,src,point_list[index]);
  canvasModified=true;
  pthread_mutex_unlock(&canvas_lock);
}


// ######################################################################
void CudaFramework::startFramework(int W,int H,int dev,MemoryPolicy mp)
{
  //Initialise
  w = W;
  h = H;
  dev1 = dev;
  mp1 = mp;
  canvas_image  = CudaImage<PixRGB<float> >(w,h,ZEROS, mp1, dev1);
  pthread_mutex_init(&canvas_lock, 0);
}

// ######################################################################
void CudaFramework::drawRectangle_centrepoint(Point2D<int> max_point, const PixRGB<float> color,const int w,const int h, const int rad)
{ //Centering around max point
  pthread_mutex_lock(&canvas_lock);
  Point2D<int> centre_point;

  centre_point.i = max_point.i - w/2;
  centre_point.j = max_point.j - h/2;
  if(centre_point.i <0 )
    centre_point.i = 0;
  if(centre_point.j <0 )
    centre_point.j = 0;

  cudaDrawRect(canvas_image,Rectangle::Rectangle(centre_point,(Dims::Dims(w,h))),color,rad);
  canvasModified=true;
  pthread_mutex_unlock(&canvas_lock);

}
// ######################################################################
void CudaFramework::drawRectangle_topleftpoint(Point2D<int> max_point, const PixRGB<float> color,const int w,const int h, const int rad)
{ //Centering around max point
  pthread_mutex_lock(&canvas_lock);
  cudaDrawRect(canvas_image,Rectangle::Rectangle(max_point,(Dims::Dims(w,h))),color,rad);
  canvasModified=true;
  pthread_mutex_unlock(&canvas_lock);

}
// ######################################################################

void CudaFramework::setText(const char* text,Point2D<int> text_coord,const PixRGB<float> color,const PixRGB<float> bgcolor,const SimpleFont& f,bool transparent_background)
{
  pthread_mutex_lock(&canvas_lock);
  writeText(text_patch,Point2D<int>(0,0),text,color,bgcolor,f,transparent_background);
  text_patch_cuda = CudaImage<PixRGB<float> >(text_patch,mp1,dev1);
  cudaInplacePaste(canvas_image,text_patch_cuda,text_coord);
  canvasModified=true;
  pthread_mutex_unlock(&canvas_lock);
}

//######################################################################
void CudaFramework::initTextLayer(int w,int h)
{
  text_patch = Image<PixRGB<float> >(w,h,NO_INIT);
}
//######################################################################
