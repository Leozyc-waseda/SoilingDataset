/*!@file BeoSub/test-PMComplex.C find pipe     */
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
// Primary maintainer for this file: Michael Montalbo <montalbo@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/test-PMComplex.C $
// $Id: test-PMComplex.C 9057 2007-11-28 04:29:48Z beobot $

#include "Component/ModelManager.H"
#include "BeoSub/BeeBrain/PreMotorComplex.H"
#include "BeoSub/BeeBrain/ComplexMovement.H"

int main(int argc, char* argv[]) {

<<<<<<< .mine
  ModelManager manager("premotor complex test");
=======

  rutz::shared_ptr<PreMotorComplex> test;
  test.reset(new PreMotorComplex("premotorcomplex"));
>>>>>>> .r9393

  nub::soft_ref<SubController> motorControl(new SubController(manager, "Controller", "PID"));
  manager.addSubComponent(motorControl);

  manager.exportOptions(MC_RECURSE);

  manager.start();

  PreMotorComplex test(motorControl, "premotorcomplex");

  rutz::shared_ptr<ComplexMovement> move_test;
  move_test.reset(new ComplexMovement());
  //move_test.functionList.push_back(NULL);
  //move_test.functionList[0] = &PreMotorComplex::turn;


  SensorInput a;
  a.data = 5.0;
  a.angle = *(new Angle(55));

  VisionInput b;
  b.position.reset(new Point3D(5, 10, 15));
  b.angle.reset(new Angle(55));

  //move_test->addOnceMove(FORWARD, 5.0, Angle(33.0));
  move_test->addMove(&PreMotorComplex::turn, ONCE, Angle(120));
  move_test->addMove(&PreMotorComplex::vis_turn, ONCE, b.position);

  //move_test->addOnceMove(&PreMotorComplex::forward, 5.0, Angle(120));

  //move_test.addRecursiveMove(&PreMotorComplex::forward, dummy4, ang);
  //printf("happens at adding function pointer18\n");

  //move_test->addOnceMove(DIVE, b.position, b.angle);
  //move_test.addOnceMove(&PreMotorComplex::vturn, NULL, b.angle);
  //move_test->addRecursiveMove(&PreMotorComplex::vturn, b.position, b.angle);

  test.run(move_test);
  //ang.setVal(-35.5);
  //c.position.x = 5555;
  //c.position.y = 5555;
  //c.position.z = 5555;
  //test.run(&move_test);
  printf("we are done\n");


  return 0;

}
