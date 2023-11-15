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

About this VirtualBox image:
============================

This VirtualBox image is derived from the Mandriva 2009 image found at
virtualbox.wordpress.com.

Upon booting up, you are automatically logged in as user 'mandriva'
with password 'reverse'. The password for root is 'toortoor'.

You may have to play with your network settings in order for them to
work and be properly configured. To do that, go to "System ->
Administration -> Configure your Computer'. Enter 'toortoor' for the
password. Then, under "Network and Internet", go to the "Network
Center". Adjust your network settings.

About the saliency algorithm:
=============================

The main program is called 'ezvision'. It is installed in
/usr/bin/ezvision

Try:

  ezvision --help

to see (a lot) of command-line options.

Have a look at:

  http://iLab.usc.edu/toolkit/screenshots.shtml

for examples of how to use the program.

Here is one to try, using a video file in /home/mandriva/Videos:

  ezvision -K --movie --in=/home/mandriva/Videos/beverly03.mpg --out=display --ior-type=None

To process your own files:
==========================

The easiest is probably to use FTP, WWW, or a USB flash disk to copy
your files to this VirtualBox image. Then process them in the
image. Finally copy the results back.

To download the source code:
============================

See http://iLab.usc.edu/toolkit/

Have a look at http://iLab.usc.edu/toolkit/ for additional
information.

Enjoy!

  -- the iLab team
