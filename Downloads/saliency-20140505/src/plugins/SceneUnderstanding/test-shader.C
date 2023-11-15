/*!@file SceneUnderstanding/test-shader.C test the opengl shader */

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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/test-viewport3D.C $
// $Id: test-viewport3D.C 13054 2010-03-26 00:12:36Z lior $



#include "GUI/ViewPort3D.H"
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

#include "GUI/XWinManaged.H"
#include "GUI/ImageDisplayStream.H"


int main(int argc, char *argv[]){

	ModelManager manager("Test Viewport");

	nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
	manager.addSubComponent(ofs);

	nub::ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
	manager.addSubComponent(ifs);


	// Parse command-line:
	if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);
	// let's get all our ModelComponent instances started:
	manager.start();

	ViewPort3D vp(320,240, false, false, false);

	//Load the shader

	vp.setCamera(Point3D<float>(0,0,100), Point3D<float>(0,0,0));

	vp.initFrame();
	Image<PixRGB<byte> > tmp(320,240,ZEROS);
	uint texId = vp.addTexture(tmp);

	GLhandleARB prog1 = vp.createShader("src/Image/OpenGlShaders/sobel.frag", GL_FRAGMENT_SHADER_ARB);
	GLhandleARB prog2 = vp.createShader("src/Image/OpenGlShaders/nonmaxsupp.frag", GL_FRAGMENT_SHADER_ARB);

	float texCoordOffsets[18];
	float xInc = 1.0/ float(320);
	float yInc = 1.0/ float(240);
	for(int i=0; i<3; i++)
		for(int j=0; j<3; j++)
		{
			texCoordOffsets[(((i*3)+j)*2)+0] = (-1.0 * xInc) + ( float(i) * xInc);
			texCoordOffsets[(((i*3)+j)*2)+1] = (-1.0 * yInc) + ( float(j) * yInc);
		}

	glUseProgramObjectARB(prog1);
	int uniformLoc = glGetUniformLocationARB(prog1, "tc_offset" );
	if( uniformLoc != -1 )
		glUniform2fvARB( uniformLoc, 9, texCoordOffsets );
	else
		LINFO("error. could not find uniformLoc for variable tc_offset");

	glUseProgramObjectARB(prog2);
	uniformLoc = glGetUniformLocationARB(prog2, "tc_offset" );
	if( uniformLoc != -1 )
		glUniform2fvARB( uniformLoc, 9, texCoordOffsets );
	else
		LINFO("error. could not find uniformLoc for variable tc_offset");

	//glUseProgramObjectARB(prog2);
	//int uniformLoc = glGetUniformLocationARB(prog2, "testValue" );

	//int i=0;
 // float val=0.6;
	while(1)
	{
		Image< PixRGB<byte> > inputImg;
		ifs->updateNext();
		const FrameState is = ifs->updateNext();
		if (is == FRAME_COMPLETE)
			break;

		//grab the images
		GenericFrame input = ifs->readFrame();
		if (!input.initialized())
			break;
		inputImg = input.asRgb();
		//i = (i+1)%90;

    Image<float> texImg = luminance(inputImg);

		vp.loadTexture(texImg, texId);
    vp.progToTexture(prog1);

		glUseProgramObjectARB(prog2);

		vp.initProjection();
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0); glVertex3f(0, 0, 0);
    glTexCoord2f(1.0, 0.0); glVertex3f(320, 0, 0);
    glTexCoord2f(1.0, 1.0); glVertex3f(320, 240 , 0);
    glTexCoord2f(0.0, 1.0); glVertex3f(0, 240, 0);
    glEnd();
    
		//Image<PixRGB<float> > img = vp.getFrame();
		Image<PixRGB<float> > img = vp.getFrameFloat();

		Image<float> tmp(img.getDims(), ZEROS);
		for(uint i=0; i < img.size(); i++)
			tmp.setVal(i, img.getVal(i).red());
    tmp *= 256;

    img = tmp;

		ofs->writeRGB(img, "output", FrameInfo("output", SRC_POS));
		usleep(10000);
	}

	manager.stop();
	exit(0);

}


