/*!@file Cell/PPU-HelloWorld.C Simple Cell processor test program */

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


// This program is a light modification of the 'simple' tutorial
// example from the IBM Cell SDK

// Original copyright:

/* --------------------------------------------------------------  */
/* (C)Copyright 2001,2006,                                         */
/* International Business Machines Corporation,                    */
/* Sony Computer Entertainment, Incorporated,                      */
/* Toshiba Corporation,                                            */
/*                                                                 */
/* All Rights Reserved.                                            */
/* --------------------------------------------------------------  */
/* PROLOG END TAG zYx                                              */

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <libspe.h>

#include "Cell/SPU_HelloWorld.h"

#define SPU_THREADS         6

int main(const int argc, const char **argv)
{
    speid_t spe_ids[SPU_THREADS];
    int i, status = 0;

    if (argc != 1) fprintf(stderr, "USAGE: %s", argv[0]);

    /* Create several SPE-threads to execute SPU_HelloWorld: */
    for (i = 0; i < SPU_THREADS; i++)
      {
        spe_ids[i] = spe_create_thread(SPE_DEF_GRP, &SPU_HelloWorld,
                                       NULL, NULL, -1, 0);
        if (spe_ids[i] == 0)
          {
            fprintf(stderr, "Failed spu_create_thread(rc=%p, errno=%d)\n",
                    spe_ids[i], errno);
            exit(1);
          }
      }

    /* Wait for SPU-thread to complete execution: */
    for (i = 0; i < SPU_THREADS; i++)
      (void)spe_wait(spe_ids[i], &status, 0);

    printf("\nThe program has successfully executed.\n");

    return (0);
}
