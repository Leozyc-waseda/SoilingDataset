/*!@file CUDA/cuda_debayer.h CUDA/GPU optimized color operations code */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/cuda_debayer.h $
// $Id: cuda_debayer.h 13228 2010-04-15 01:49:10Z itti $
//
#ifndef CUDA_DEBAYER_H_DEFINED
#define CUDA_DEBAYER_H_DEFINED

#include <cuda.h>
#include "CUDA/cutil.h"
#include "cudadefs.h"



//######################################################################
__global__ void cuda_debayer_kernel_Nearest_Neighbour(float *src,float3_t  *dptr,int w,int h,int tile_width,int tile_height)
{

  int Row = blockIdx.y * tile_height + (threadIdx.y);
  int Col = blockIdx.x * tile_width + (threadIdx.x);
  if((Row*w+Col)<w*h)
    {
      if(Row%2==0 && Col%2==0)
        {
          dptr[Row*w+Col].p[0]= src[ Col + ((Row+1) * w) ];
          dptr[Row*w+Col].p[1]= src[ Col + ( (Row) * w ) ];
          dptr[Row*w+Col].p[2]= src[ Col + 1 + Row * w ];
        }
      if(Row%2==0 && Col%2==1)
        {
          dptr[Row*w+Col].p[0]= src[Col -1 + ( (Row+1) * w ) ];
          dptr[Row*w+Col].p[1]= src[Col + (Row+1) * w];
          dptr[Row*w+Col].p[2]= src[Col  + ( (Row) * w ) ];
        }
      if(Row%2==1 && Col%2==0)
        {
          dptr[Row*w+Col].p[0]= src[Col  + (Row * w)  ];
          dptr[Row*w+Col].p[1]= src[Col +  Row * w ];
          dptr[Row*w+Col].p[2]= src[Col +1 + ( (Row-1) * w) ];
        }
      if(Row%2==1 && Col%2==1)
        {
          dptr[Row*w+Col].p[0]= src[Col -1 + Row * w ];
          dptr[Row*w+Col].p[1]= src[Col  + Row * w];
          dptr[Row*w+Col].p[2]= src[Col  + ( (Row-1) * w) ];
        }
    }

}

//######################################################################
__global__ void cuda_debayer_kernel_MHC_optimised(float *src,float3_t  *dptr,int w,int h,int tile_width,int tile_height)
{
  //Create the shared memory space
  //float *Mds = (float*) shared_data;
  //Calculate necessary indexes
  //Create border
  /*
  float *border = (float *) &data[tile_width];
  // Load the border always and this will speed up the computation
  // and get rid of all this symbol table crap

  //Loading the border
  int width_border = tile_width + 4;
  int height_border = tile_height + 4;
  int shared_index = sy*width_border+sx; //How to adjust the border


  Mds[shared_index]=src[global_index];

  */

  const int sts_y = blockIdx.y * tile_height;
  const int sts_x = blockIdx.x * tile_width;

  const int sx = threadIdx.x;                          // x source pixel within source tile
  const int sy = threadIdx.y;                           // y source pixel within source tile
  const int Row = sts_y + sy;
  const int Col = sts_x + sx;


   float *data = (float*) shared_data;


  //In the shared memory

   data[(sy*tile_width)+sx]=src[(Row*w)+Col];

   __syncthreads();
  if(Row<h&&Col<w)
     {
       //Threads synced

       //Global row column conditions
       /*
       int Row_min_1=Row-1;
       int Row_min_2=Row-2;
       int Row_pls_1=Row+1;
       int Row_pls_2=Row+2;
       int Col_min_1=Col-1;
       int Col_min_2=Col-2;
       int Col_pls_1=Col+1;
       int Col_pls_2=Col+2;
       int sx_min_1=sx-1;
       int sx_min_2=sx-2;
       int sx_pls_1=sx+1;
       int sx_pls_2=sx+2;
       int sy_min_1=sy-1;
       int sy_min_2=sy-2;
       int sy_pls_1=sy+1;
       int sy_pls_2=sy+2;

       bool bulk = (sy>1)&&(sy<tile_height-2)&&(sx>1)&&(sx<tile_width-2);
       */
       /*
       if(!bulk)
         {
           if(sts_y<2||sts_y>h-3)
             {
               switch(sts_y) //Global row
                 {
                 case 0: Row_min_1=Row; Row_min_2 = Row; break;
                 case 1: Row_min_2=Row;break;

                 }
               switch(sts_y-h) //Global row
                 {
                 case -2: Row_pls_2 =Row;break;
                 case -1: Row_pls_2= Row; Row_pls_1 =Row;break;
                 }
             }
           if(sts_x<2||sts_x>w-3)
             {
               switch(sts_x) //Global column
                 {
                 case 0: Col_min_1=Col; Col_min_2 = Col; break;
                 case 1: Col_min_2=Col;break;

                 }
               switch(sts_x-w) //Global row
                 {
                 case -2: Col_pls_2 =Col;break;
                 case -1: Col_pls_2= Col; Col_pls_1 =Col;break;
                 }
             }
           //Local row and column conditions

           if(sy<2||sy>tile_height-3)
             {
               switch(sy) //Local row sy
                 {
                 case 0: sy_min_1 = tile_height+2; sy_min_2 = tile_height+3; break;
                 case 1: sy_min_2 =tile_height+2;break;


                 }
               switch(sy-tile_height)
                 {
                 case -2:sy_pls_2=tile_height;break;
                 case -1:sy_pls_1=tile_height;sy_pls_2=tile_height+1;break;
                 }
             }
           if(sx<2||sx>tile_width-3)
             {
               switch(sx) //Local column sx
                 {
                 case 0: sx_min_1 = tile_width+2; sx_min_2 = tile_width+3; break;
                 case 1: sx_min_2 =tile_width+2;break;



                 }
               switch(sx-tile_width)
                 {
                 case -2:sx_pls_2=tile_width;break;
                 case -1:sx_pls_1=tile_width;sx_pls_2=tile_width+1;break;
                 }
             }
           //Assigning new values at those border points
           if(sy<2||sy>tile_height-3)
             {
               switch(sy)
                 {
                 case 0:
                   {
                     Mds[sy_min_1*tile_width+sx_min_1] = src[Col_min_1 + ((Row_min_1) * w)];
                     Mds[sy_min_1*tile_width+sx_pls_1] = src[Col_pls_1 + ((Row_min_1) * w)];
                     Mds[sy_min_1*tile_width+sx] = src[Col + ((Row_min_1) * w)];
                     Mds[sy_min_2*tile_width+sx] = src[Col + ((Row_min_2) * w)];break;
                   }
                 case 1:
                   Mds[sy_min_2*tile_width+sx] = src[Col + ((Row_min_2) * w)]; break;

                 }
               switch(sy-tile_height)
                 {   case -2:
                     Mds[sy_pls_2*tile_width+sx] =  src[Col + ((Row_pls_2) * w)]; break;
                 case -1:
                   {
                     Mds[sy_pls_1*tile_width+sx_min_1]  =  src[Col_min_1 + ((Row_pls_1) * w)];
                     Mds[sy_pls_1*tile_width+sx_pls_1]  =  src[Col_pls_1 + ((Row_pls_1) * w)];
                     Mds[sy_pls_1*tile_width+sx] =  src[Col + ((Row_pls_1) * w)];
                     Mds[sy_pls_2*tile_width+sx] =  src[Col + ((Row_pls_2) * w)]; break;
                   }
                 }
             }
           if(sx<2||sx>tile_width-3)
             {
               switch(sx)
                 {
                 case 0:
                   {
                     Mds[sy_min_1*tile_width+sx_min_1]  = src[Col_min_1 + ((Row_min_1) * w)];
                     Mds[sy_pls_1*tile_width+sx_min_1]  = src[Col_min_1 + ((Row_pls_1) * w)];
                     Mds[sy*tile_width+sx_min_1]  = src[Col_min_1 + ((Row) * w)];
                     Mds[sy*tile_width+sx_min_2] = src[Col_min_2 + ((Row) * w)]; break;
                   }
                 case 1:
                   Mds[sy*tile_width+sx_min_2]  = src[Col_min_2 + ((Row) * w)]; break;

                 }
               switch(sx-tile_width)
                 {
                 case -2:
                   Mds[sy*tile_width+sx_pls_2]  =  src[Col_pls_2 + ((Row) * w)]; break;
                 case -1:
                   {
                     Mds[sy_min_1*tile_width+sx_pls_1]  =  src[Col_pls_1 + ((Row_min_1) * w)];
                     Mds[sy_pls_1*tile_width+sx_pls_1]  =  src[Col_pls_1 + ((Row_pls_1) * w)];
                     Mds[sy*tile_width+sx_pls_1]  =  src[Col_pls_1 + ((Row) * w)];
                     Mds[sy*tile_width+sx_pls_2]  =  src[Col_pls_2 + ((Row) * w)]; break;
                   }
                 }
             }
         }*/
       //Calculating constants

         {
       int A,B,C,D,E,F;

       A = data[(sy-1)*(tile_width)+sx-1]+
         data[(sy-1)*(tile_width)+sx+1]+
         data[(sy+1)*(tile_width)+sx-1]+
         data[(sy+1)*(tile_width)+sx+1];


       B = data[(sy-1)*(tile_width)+sx]+
         data[(sy+1)*(tile_width)+sx];


       C = data[(sy-2)*(tile_width)+sx]+
         data[(sy+2)*(tile_width)+sx];

       D = data[(sy)*(tile_width)+sx-1]+
         data[(sy)*(tile_width)+sx+1];

       E = data[(sy)*(tile_width)+sx-2]+
         data[(sy)*(tile_width)+sx+2];

       F=   data[(sy)*(tile_width)+sx];

 /*
       A = src[(sy_min_1)*(tile_width)+sx_min_1]+
         src[(sy_min_1)*(tile_width)+sx_pls_1]+
         src[(sy_pls_1)*(tile_width)+sx_min_1]+
         src[(sy_pls_1)*(tile_width)+sx_pls_1];


       B = src[(sy_min_1)*(tile_width)+sx]+
         src[(sy_pls_1)*(tile_width)+sx];


       C = src[(sy_min_2)*(tile_width)+sx]+
         src[(sy_pls_2)*(tile_width)+sx];

       D = src[(sy)*(tile_width)+sx_min_1]+
         src[(sy)*(tile_width)+sx_pls_1];

       E = src[(sy)*(tile_width)+sx_min_2]+
         src[(sy)*(tile_width)+sx_pls_2];

       F=   src[(sy)*(tile_width)+sx];

   *//*
       A = src[(Row_min_1)*(w)+Col_min_1]+
         src[(Row_min_1)*(w)+Col_pls_1]+
         src[(Row_pls_1)*(w)+Col_min_1]+
         src[(Row_pls_1)*(w)+Col_pls_1];


       B = src[(Row_min_1)*(w)+Col]+
         src[(Row_pls_1)*(w)+Col];


       C = src[(Row_min_2)*(w)+Col]+
         src[(Row_pls_2)*(w)+Col];

       D = src[(Row)*(w)+Col_min_1]+
         src[(Row)*(w)+Col_pls_1];

       E = src[(Row)*(w)+Col_min_2]+
         src[(Row)*(w)+Col_pls_2];

       F=   src[(Row)*(w)+Col];
     */
       //Odd Even Row Col conditions
       /* int sit_3;
          if(Row%2==0)
          sit_3 = 0;
          else
          sit_3 = 1;
          if(Col%2==0)
          sit_3 = sit_3 + 2;
          else
          sit_3 = sit_3 + 4;*/
       if(Row%2==0 && Col%2==0)
         {
           dptr[Row*w+Col].p[0]=  ((5*F) -(1 * (A + C)) + (4 * B) + ((1/2) * E))/8 ;
           dptr[Row*w+Col].p[1]=   F;
           dptr[Row*w+Col].p[2]=  ((5*F) -(1 * (A + E)) + (4 * D) + ((1/2) * C))/8 ;

         }
        if(Row%2==1 && Col%2==0){
         dptr[Row*w+Col].p[0]=  F ;
         dptr[Row*w+Col].p[1]=  ((4*F) -(1 * (C + E)) + (2 * (D+B)))/8;
         dptr[Row*w+Col].p[2]=  ((6*F) -((3/2) * (E + C)) + (2 * A)) /8 ;

       }
       if(Row%2==0 && Col%2==1){
         dptr[Row*w+Col].p[0]=  ((6*F) -((3/2) * (E + C)) + (2 * A)) /8 ;
         dptr[Row*w+Col].p[1]=  ((4*F) -(1 * (C + E)) + (2 * (D+B)))/8;
         dptr[Row*w+Col].p[2]=  F ;
       }
       if(Row%2==1 && Col%2==1){
         dptr[Row*w+Col].p[0]=  ((5*F) -(1 * (A + E)) + (4 * D) + ((1/2) * C))/8 ;
         dptr[Row*w+Col].p[1]=   F;
         dptr[Row*w+Col].p[2]=  ((5*F) -(1 * (A + C)) + (4 * B) + ((1/2) * E))/8;
       }
         }
       /*
         switch(sit_3)
         {
         case 2: dptr[Row*w+Col].p[0]=  ((5*F) -(1 * (A + C)) + (4 * B) + ((1/2) * E))/8 ;
         dptr[Row*w+Col].p[1]=   F;
         dptr[Row*w+Col].p[2]=  ((5*F) -(1 * (A + E)) + (4 * D) + ((1/2) * C))/8 ;
         break;
         case 3: dptr[Row*w+Col].p[0]=  F ;
         dptr[Row*w+Col].p[1]=  ((4*F) -(1 * (C + E)) + (2 * (D+B)))/8;
         dptr[Row*w+Col].p[2]=  ((6*F) -((3/2) * (E + C)) + (2 * A)) /8 ;

         break;
         case 4: dptr[Row*w+Col].p[0]=  ((6*F) -((3/2) * (E + C)) + (2 * A)) /8 ;
         dptr[Row*w+Col].p[1]=  ((4*F) -(1 * (C + E)) + (2 * (D+B)))/8;
         dptr[Row*w+Col].p[2]=  F ;
         break;
         case 5: dptr[Row*w+Col].p[0]=  ((5*F) -(1 * (A + E)) + (4 * D) + ((1/2) * C))/8 ;
         dptr[Row*w+Col].p[1]=   F;
         dptr[Row*w+Col].p[2]=  ((5*F) -(1 * (A + C)) + (4 * B) + ((1/2) * E))/8;
         break;

         } */
     }

}


__global__ void cuda_kernel_debayer(float *src,float3_t  *dptr,int w,int h,int tile_width,int tile_height)
{


    int Row = blockIdx.y * tile_height + (threadIdx.y);
    int Col = blockIdx.x * tile_width  + (threadIdx.x);
    int A,B,C,D,E,F;


  if((Row>1||Row<(w-2))&&(Col>1||Col<(h-2))&&((Row*w+Col)<=w*h))
    {
      A =         src[ Col-1 + ((Row-1) * w) ]+
            src[ Col+1 + ((Row-1) * w) ]+
            src[ Col-1 + ((Row+1) * w) ]+
            src[ Col+1 + ((Row+1) * w) ];

          B =     src[ Col + ((Row+1) * w) ] +
            src[ Col + ((Row-1) * w) ];

          C=      src[ Col + ((Row+2) * w) ]+
            src[ Col + ((Row-2) * w) ];

          D=      src[ Col-1 + ((Row) * w) ]+
            src[ Col+1 + ((Row) * w) ];

          E=      src[ Col-2 + ((Row) * w) ]+
            src[ Col+2 + ((Row) * w) ];

          F=      src[ Col + ((Row) * w) ];

      if(Row%2==0 && Col%2==0)
        {


          dptr[Row*w+Col].p[0]=  ((5*F) -(1 * (A + C)) + (4 * B) + ((1/2) * E))/8 ;
          dptr[Row*w+Col].p[1]=   F;
          dptr[Row*w+Col].p[2]=  ((5*F) -(1 * (A + E)) + (4 * D) + ((1/2) * C))/8 ;
        }
      if(Row%2==0 && Col%2==1)
        {


          dptr[Row*w+Col].p[0]=  ((6*F) -((3/2) * (E + C)) + (2 * A)) /8 ;
          dptr[Row*w+Col].p[1]=  ((4*F) -(1 * (C + E)) + (2 * (D+B)))/8;
          dptr[Row*w+Col].p[2]=  F ;
        }
      if(Row%2==1 && Col%2==0)
        {



          dptr[Row*w+Col].p[0]=  F ;
          dptr[Row*w+Col].p[1]=  ((4*F) -(1 * (C + E)) + (2 * (D+B)))/8;
          dptr[Row*w+Col].p[2]=  ((6*F) -((3/2) * (E + C)) + (2 * A)) /8 ;

        }
      if(Row%2==1 && Col%2==1)
        {


          dptr[Row*w+Col].p[0]=  ((5*F) -(1 * (A + E)) + (4 * D) + ((1/2) * C))/8 ;
          dptr[Row*w+Col].p[1]=   F;
          dptr[Row*w+Col].p[2]=  ((5*F) -(1 * (A + C)) + (4 * B) + ((1/2) * E))/8;
        }

       if (dptr[Row*w+Col].p[0] < 0)
                dptr[Row*w+Col].p[0] = 0;
       else if(dptr[Row*w+Col].p[0] > 255)
                dptr[Row*w+Col].p[0] = 255;
       if (dptr[Row*w+Col].p[1] < 0)
                dptr[Row*w+Col].p[1] = 0;
       else if(dptr[Row*w+Col].p[1] > 255)
                dptr[Row*w+Col].p[1] = 255;
        if (dptr[Row*w+Col].p[2] < 0)
                dptr[Row*w+Col].p[2] = 0;
       else if(dptr[Row*w+Col].p[2] > 255)
                dptr[Row*w+Col].p[2] = 255;
    }
}

#endif

