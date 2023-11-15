/*!@file AppMedia/app-omnicamera.C simple program to exercise FrameIstream
  and FrameOstream */

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
// Primary maintainer for this file: Chin-Kai Chang <chinkaic@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/app-omnicamera.C $
// $Id: app-stream.C 9841 2008-06-20 22:05:12Z kai$
//

#ifndef APPMEDIA_APP_OMNICAMERA_C_DEFINED
#define APPMEDIA_APP_OMNICAMERA_C_DEFINED

#include "Component/ModelManager.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameIstream.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"
#include "Image/CutPaste.H"     // for inplacePaste()
#include "Image/DrawOps.H" //for writeText and SimpleFont
#include "Image/ColorOps.H" //for luminance
#include "Image/MathOps.H" //for inplaceNormalize() 
#include "Image/Normalize.H" //for inplaceNormalize() 
#include "GUI/ImageDisplayStream.H"
#include <cstdio>//for sprintf
#include "Gist/VanishingPointDetector.H" 
#include "Util/Timer.H"
#include "GUI/XWinManaged.H"


Image<PixRGB<byte> > toColor(Image<float> input)
{
				return toRGB(normalizeFloat(input, true));
}
Point2D<int> handleUserEvent(nub::soft_ref<OutputFrameSeries> ofs,Image<PixRGB<byte> >dispImg)
{
				//handle clicks
				const nub::soft_ref<ImageDisplayStream> ids =
								ofs->findFrameDestType<ImageDisplayStream>();

				const rutz::shared_ptr<XWinManaged> uiwin =
								ids.is_valid()
								? ids->getWindow("Omni-camera")
								: rutz::shared_ptr<XWinManaged>();
				Point2D<int> mouseClick(-1,-1);
				if (uiwin.is_valid())
				{
								int key = uiwin->getLastKeyPress();

								switch(key)
								{
												case -1:
																break;
												case 13: //g (MAC)
												case 42: //g label ground
																break;
												case 11: //f (mac)
												case 41: //f forward faster
																break;
												case 10: //d (mac) 
												case 40: //d forward slower	
																break;
												case 9:  //s (mac) 
												case 39: //s save screenshot 
																break;
												case  8: //a (mac) 
												case 38: //a forward normal
																break;
												case 12: //h (mac) 
												case 43: //h switch labeling ground truth mode 
																break;
																//case 12: //j (mac) 
												case 44: //j play only ground truth image
																break;
												case 57: //space (mac)(n on PC) 
												case 65: //space pause/resume
																break;
												case 14: //z (mac)
												case 52: //z switch to Anf
																break;
												case 15: //x switch to SSL 
												case 53: //x switch to SSL 
																break;
												case 16: //c (mac) 
												case 54: //c switch to Equad
																break;
												case 17: //v (mac)
												case 55: //v switch to HNB 
																break;
												case 19: //b (mac) 
												case 56: //b switch to RoadGenerator 
																break;
												case 20: //q (mac) 
												case 24: //q Turn on/off VPD 
																break;
												case 21: //w (mac) 
												case 25: //w Turn on/off CSH 
																break;
												case 111://up key
																break;
												case 116://down key
																break;
												case 113://left key
																break;
												case 114://right key
																break;


												default:		
																LINFO("key %i", key);
																break;
								}

								Point2D<int> pos = uiwin->getLastMouseClick();
								if (pos.isValid())
								{
												PixRGB<byte> pixColor = dispImg.getVal(pos);
												LINFO("Mouse Click (%d %d) Color (%d,%d,%d)", 
																				pos.i,pos.j,pixColor.red(),pixColor.green(),pixColor.blue());
												mouseClick = pos;
								} 
				}
	return mouseClick;
}	
Image<PixRGB<byte> >crop(Image<PixRGB<byte> > input,Point2D<int> center,float radius,Dims d)
{
//because center is in rescaled image,find mapped loc first;
int ow = input.getWidth();
int oh = input.getHeight();
int   newx = int((float)center.i/(float)d.w()*ow);
int   newy = int((float)center.j/(float)d.h()*oh);
float newr = (radius/(float)d.w())*ow +5.0;//add 5 pixel on the border
//LINFO("ow %d,oh %d, nx %d, ny %d, nr %f",ow,oh,newx,newy,newr);
//new image is a square
int w = 2*newr;
int h = 2*newr;
Image<PixRGB<byte> > newImg(w,h,ZEROS);

for(int i = 0;i< w;i++)
	for(int j = 0;j<h;j++){
		int ix = newx - newr + i;//from center shift to top left
		int iy = newy - newr + j;
		//LINFO("ix %d,iy %d",ix,iy);
		if(input.coordsOk(ix,iy))
						newImg.setVal(i,j,input.getVal(ix,iy));
	}

return newImg;

}
// ######################################################################
Image<PixRGB<byte> >unwrapping(Image<PixRGB<byte> > input,float off)
{
int ow = input.getWidth();
int oh = input.getHeight();
float radius = ow/2.0;
float nw = 2.0*M_PI*radius;
int c = (int)nw;
int ri = (int)radius;
Point2D<int>center((int)radius,(int)radius);

Image<PixRGB<byte> > uwImg((int)nw,(int)radius,ZEROS);

for(int i = 0;i<c;++i)
{
	float offset = M_PI/2;
	float tha = ((float)i/(float)c) * 2 * M_PI +offset+off;//radius
  float cix,ciy;
	for(int rr = 0;rr < ri; rr++){
		cix = center.i + rr*sin(tha);  
		ciy = center.j - rr*cos(tha);
		if(uwImg.coordsOk(i,rr))
		{
						uwImg.setVal(i,rr,input.getVal((int)cix,(int)ciy));
		}
	}



//	cix = center.i + radius*sin(tha);  
//	ciy = center.j - radius*cos(tha);
//
//
//
//
//	//map pixel from (cix,ciy) to center
//  //LINFO("i %d,tha %f,ci(%d,%d)",i,tha/M_PI*180,(int)cix,(int)ciy);
//  int dx = center.i - cix, ax = abs(dx) << 1, sx = signOf(dx);
//  int dy = center.j - ciy, ay = abs(dy) << 1, sy = signOf(dy);
//  int x = center.i, y = center.j;
//	//LINFO("dxy(%d,%d) sxy(%d,%d) xy(%d,%d)",dx,dy,sx,sy,x,y);
//  const int w = ow; 
//  const int h = oh;
//  int count = 0;
//	if (ax > ay)
//	{
//					int d = ay - (ax >> 1);
//					for (;;)
//					{
//							//`		Raster::waitForKey();
//									int uyi = count;
//									if (x >= 0 && x < w && y >= 0 && y < h)
//									{
//										if(uwImg.coordsOk(i,uyi))
//												{
//													uwImg.setVal(i,uyi,input.getVal(x,y));
//									//				LINFO(">x %d-> %d, y %d -> %d",x,i,y,uyi);
//												}
//									}
//									//LINFO(">In xy (%d,%d) d %d",x,y,d);										
//									if (uyi > ri) break;
//									if (d >= 0) { y += sy; d -= ax; }
//									x += sx; d += ay; count++;
//					}
//	}
//	else
//	{
//					int d = ax - (ay >> 1);
//					for (;;)
//					{
//									//Raster::waitForKey();
//									int uyi = count;
//									if (x >= 0 && x < w && y >= 0 && y < h)
//									{
//										if(uwImg.coordsOk(i,uyi))
//										{
//														uwImg.setVal(i,uyi,input.getVal(x,y));
//														//LINFO("<x %d-> %d, y %d -> %d",x,i,y,uyi);
//										}
//									}
//									//LINFO("<In xy (%d,%d) d %d",x,y,d);										
//									if (uyi > ri) break;
//									if (d >= 0) { x += sx; d -= ay; }
//									y += sy; d += ax; count++;
//					}
//	}
}
//fold image to two raw
int c2 = c/2;
Image<PixRGB<byte> > uwImg2(c2,oh,ZEROS);
for(int x = 0;x < c2;x++)
	for(int y = 0;y < oh;y++)
	 uwImg2.setVal(x,y,uwImg.getVal(x+c2*(y/ri),y%ri));

return uwImg2;
}
int main(int argc, char **argv)
{

	// instantiate a model manager:
	ModelManager manager("Omni-camera");

	// Instantiate our various ModelComponents:

	nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
	manager.addSubComponent(ifs);

	nub::soft_ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
	manager.addSubComponent(ofs);

	manager.exportOptions(MC_RECURSE);

	// Parse command-line:
	if (manager.parseCommandLine(argc, argv, "[pyr_size] [angle] [min_size]", 0, 4) == false) return(1);


	// let's do it!
	manager.start();
	bool keepGoing = true; uint index = 0;
	Timer timer(1000000); //float time = 0.0;
  int state = 0;
	float radius = 164.0;
	//float radius = 0.0;
	//Point2D<int> center(-1,-1);
	Point2D<int> center(342,179);
	float offset = 0.0;
	while(keepGoing)
	{
		ifs->updateNext();
		Image<PixRGB<byte> > ima = ifs->readRGB();

		if(!ima.initialized()) { keepGoing = false; }
		else
		{
			uint ow = ima.getWidth();
			uint oh = ima.getHeight();
			uint w = 640; 
			uint h = oh*w/ow;

			Image<PixRGB<byte> > preview = rescale(ima,Dims(w,h));

			Image<PixRGB<byte> > dispIma(w*2,h , NO_INIT);

			timer.reset();
			inplacePaste(dispIma, preview, Point2D<int>(0, 0));
			inplacePaste(dispIma, preview, Point2D<int>(w, 0));


			Point2D<int> mpos = handleUserEvent(ofs,dispIma);
			if(mpos.i != -1 && mpos.j != -1 && radius ==0.0)
			{
				//Get first click, set center location
				if(state == 0){
					LINFO("Set Center to (%d,%d)",mpos.i,mpos.j);
					center = mpos;
					state = 1;
			 //Get sencode click, calculate radius
				}else if(state == 1){
					int dx = mpos.i - center.i;
					int dy = mpos.j - center.j;
					float distance = sqrt(dx*dx+dy*dy);
					radius = distance;
					LINFO("Click (%d,%d),Radius is %f",mpos.i,mpos.j,radius);
					state = 2;
				}
			}
			
			if(state ==2||radius !=0.0){
				Image<PixRGB<byte> > omni = crop(ima,center,radius,Dims(w,h));
				offset+= 1.0/180*M_PI;
				if(offset>2*M_PI) offset = 0.0;
				Image<PixRGB<byte> > uw   = unwrapping(omni,offset);

				Image<PixRGB<byte> > cropIma (omni.getWidth()+uw.getWidth(),omni.getHeight() , ZEROS);
				inplacePaste(cropIma, omni, Point2D<int>(0, 0));
				inplacePaste(cropIma, uw, Point2D<int>(omni.getWidth(), 0));
				ofs->writeRGB(cropIma, "Omni-camera");

			}else{
				ofs->writeRGB(dispIma, "Omni-camera");
			}
			ofs->updateNext();

		}
		index++;
	}
	Raster::waitForKey();


	return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif // APPMEDIA_APP_STREAM_C_DEFINED
