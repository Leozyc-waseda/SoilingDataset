/*!@file CUDA/CudaFramework.H A class for CUDA display framework  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaFramework.H $
// $Id: CudaFramework.H 12962 2010-03-06 02:13:53Z irock $
//
#ifndef CUDAFRAMEWORK_H_DEFINED
#define CUDAFRAMEWORK_H_DEFINED

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <errno.h>
#include <GL/glew.h>
#include <cuda_runtime.h>
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
#include "CUDA/CudaShapeOps.H"
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
#define MAX_SIZE 14

class CudaFramework
{

public:


  //! Constructor
  CudaFramework();


  //! Destructor
  ~CudaFramework();


  //! Get Canvas Width
  int getCanvasW();


  //! Get Canvas Height
  int getCanvasH();


  //! Get the canvas
  CudaImage<PixRGB<float> >& getCanvas();


  //! Get point with index i
  Point2D<int> getPoint(int i);


  //! Get the Memory policy
  MemoryPolicy getMP();


  //! Get the device
  int getDev();


  //! Get the lock Attributes
  int getLockAtt();


  //! Get if canvas is modified
  bool getCanvasModified();


  //! Get image total width
  int getW();


  //! Get Image total height
  int getH();


  //! Sets the memory policy
  void setMP(MemoryPolicy mp);


  //! Sets the device
  void setDev(int dev);


  //! Sets the pthread mutex to destroy
  void setMutexDestroy();


  //! Sets the pthread mutex to unlock
  void setMutexUnlock();


  //! Sets the canvas modified to new state
  void setCanvasModified(bool new_state);


  //! Sets the Point with index i to coordinates in pt
  void setPoint(int i,Point2D<int> pt);


  //! Sets the total width as W
  void setW(int W);


  //! Sets the total height as H
  void setH(int H);


  //! Updates the canvas with index i and image stored in src
  void updateCanvas(int i, CudaImage<PixRGB<float> > &src);


  //! Starts the framework with width w , height h , cuda device dev and memory policy mp
  void startFramework(int w,int h,int dev,MemoryPolicy mp);

  //! Draws a rectangle on the canvas with centre point as input
  void drawRectangle_centrepoint(Point2D<int> centre_point, const PixRGB<float> color,const int w,const int h, const int rad);

  //! Draws a rectangle on the canvas with top left point as input
  void drawRectangle_topleftpoint(Point2D<int> topleft_point, const PixRGB<float> color,const int w,const int h, const int rad);

  //! Sets the text on the text patch which is in place pasted on the canvas
  void setText(const char* text,Point2D<int> text_coord,const PixRGB<float> color,const PixRGB<float> bgcolor,const SimpleFont& f,bool transparent_background);

  //! Initialises the text patch with width w and height h
  void initTextLayer(int w,int h);

private:

  Point2D<int> point_list[MAX_SIZE];
  int dev1;
  MemoryPolicy mp1;
  int w,h;
  CudaImage<PixRGB<float> > canvas_image;
  pthread_mutex_t canvas_lock;
  bool canvasModified;
  int saliency_rectangle_size ;
  Image<PixRGB<float> > text_patch;
  CudaImage<PixRGB<float> > text_patch_cuda;
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */


