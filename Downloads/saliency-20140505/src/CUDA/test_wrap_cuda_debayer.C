#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/PyramidOps.H"
#include "Image/LowPass.H"
#include "Image/Pixels.H"
#include "Image/ColorOps.H"
#include "Image/MathOps.H"
#include "Raster/Raster.H"
#include "CUDA/CudaImage.H"

#include "wrap_c_cuda.h"
#include "CUDA/cutil.h"
#include "CUDA/CudaColorOps.H"
#include "CUDA/CudaLowPass.H"
#include "CUDA/CudaDevices.H"
#include "CUDA/CudaImageSet.H"
#include "CUDA/CudaImageSetOps.H"
#include "CUDA/CudaPyramidOps.H"
#include "CUDA/CUDAdebayer.H"
#include "Util/fpu.H"
#include <cmath>
#include <fstream>
#include <sys/time.h>

// Allowance for one particular pixel
#define ROUNDOFF_MARGIN 0.00005//0.00005
// Allowance for image wide bias
#define BIAS_MARGIN     0.000015//0.000001

// ######################################################################
void compareregions(const Image<float> &c, const Image<float> &g, const uint rowStart, const uint rowStop, const uint colStart, const uint colStop)
{
  uint w,h;
  w = c.getWidth();
  h = c.getHeight();
  if(w != (uint) g.getWidth() || h != (uint) g.getHeight())
  {
    LINFO("Images are not the same size");
    return;
  }
  if(rowStart > rowStop || colStart > colStop || rowStop > h || colStop > w)
  {
    LINFO("Invalid regions to compare");
    return;
  }
  for(uint j=rowStart;j<rowStop;j++)
  {
    printf("\nC[%d]: ",j);
    for(uint i=colStart;i<colStop;i++)
    {
      printf("%.3f ",c.getVal(i,j));
    }
    printf("\nG[%d]: ",j);
    for(uint i=colStart;i<colStop;i++)
    {
      printf("%.3f ",g.getVal(i,j));
    }
  }
  printf("\n");

}

void calculateCudaSaliency(const CudaImage<PixRGB<float> > &input, const int cudaDeviceNum)
{
  // Copy the input image to the CUDA device
  CudaImage<PixRGB<float> > cudainput = CudaImage<PixRGB<float> >(input,GLOBAL_DEVICE_MEMORY,cudaDeviceNum);
  CudaImage<float> red, green, blue;
  // Get the components
  cudaGetComponents(cudainput,red,green,blue);
  // Get the luminance
  CudaImage<float> lumin = cudaLuminance(cudainput);
}


void printImage(const Image<float> &in)
{
  int cnt=0;
  for(int i=0;i<in.getWidth();i++)
  {
    for(int j=0;j<in.getHeight();j++)
    {
      printf("%g ",in.getVal(i,j));
      cnt++;
      if(cnt==30)
      {
        cnt=0;
      }
    }
    printf("\n");
  }
}

void printImages(const Image<float> &in1, const Image<float> &in2)
{
  int w,h;
  w = in1.getWidth(); h = in2.getHeight();
  if(in2.getWidth() != w || in2.getHeight() != h)
    LFATAL("Cannot compare two different image sizes im1[%dx%d] im2[%dx%d]",
           in1.getWidth(),in1.getHeight(),in2.getWidth(),in2.getHeight());
  for(int i=0;i<in1.getWidth();i++)
  {
    for(int j=0;j<in1.getHeight();j++)
    {
      printf("At [%d,%d] Im1 = %f Im2 = %f\n",i,j,in1.getVal(i,j),in2.getVal(i,j));
    }
  }
}


void testDiff(const Image<float> in1, const Image<float> in2)
{
  Image<float> diff = in1-in2;
  Point2D<int> p;
  float mi, ma, av;
  int w,h;
  w = in1.getWidth(); h = in2.getHeight();
  if(in2.getWidth() != w || in2.getHeight() != h)
    LFATAL("Cannot compare two different image sizes im1[%dx%d] im2[%dx%d]",
           in1.getWidth(),in1.getHeight(),in2.getWidth(),in2.getHeight());
  getMinMaxAvg(diff,mi,ma,av);
  bool acceptable = mi> -ROUNDOFF_MARGIN && ma< ROUNDOFF_MARGIN && std::abs(av) < BIAS_MARGIN;
  LINFO("%s: %ux%u image, #1 - #2: avg=%f, diff = [%f .. %f]",
        mi == ma && ma == 0.0F ? "MATCH" : (acceptable ? "ACCEPT" : "FAIL"), w, h, av, mi, ma);
  if(!acceptable)//(mi != ma || ma != 0.0F)
  {
    getMinMaxAvg(in1,mi,ma,av);
    LINFO("Image 1 [%dx%d]: avg=%f, diff = [%f .. %f]",
          w, h, av, mi, ma);
    getMinMaxAvg(in2,mi,ma,av);
    LINFO("Image 2 [%dx%d]: avg=%f, diff = [%f .. %f]",
          w, h, av, mi, ma);
    findMax(diff,p,ma);
    LINFO("Maximum difference %f is located at %dx%d",ma,p.i,p.j);
    compareregions(in1,in2,std::max(0,p.j-5),std::min(h,p.j+5),std::max(0,p.i-5),std::min(w,p.i+5));
  }
  //printImage(diff);
}
void testdiff_bayer(const Image<PixRGB<float> > img1,const Image<PixRGB<float> > img2)
{
        //Compare the 2 images pixel by pixel
        Image<PixRGB<float> >::const_iterator img1_ptr = img1.begin();
        Image<PixRGB<float> >::const_iterator img2_ptr = img2.begin();
        Image<PixRGB<float> >::const_iterator stop  = img2.end();

        int i,counter=0;
        while(img2_ptr!=stop)
        {
                 counter++;
                for(i=0;i<3;i++)
                        printf("Counter:%d  Image1:%f || Image2:%f \n",counter, img1_ptr->p[i], img2_ptr->p[i]);

                getchar();

                ++img1_ptr;

                ++img2_ptr;
        }
}/*
Image<PixRGB<float> > makeR(const Image<float>& img)
{
        Image<PixRGB<float> > result(img.getDims(),NO_INIT);

        Image<PixRGB<float> >::iterator start = result.beginw();
        Image<PixRGB<float> >::iterator end = result.endw();
        int i=0;
        int j=0;
        while(start!=end)
        {
                          start->setRed(img.getVal(j,i));

                          start->setGreen(0);

                          start->setBlue(0);
                        start++;
                        j++;

                        if(j==result.getWidth())
                        {
                                i++;
                                j=0;
                        }
                        if(i==result.getWidth())
                        {

                                i=i-1;

        }


        //Iterate over  and set values R,0,0
        for(int i=0;i<result.getWidth();i++)
        {
                for(int j=0;j<result.getHeight();j++)
                {      // printf("i:%d  j:%d",i,j);
                        //start[i+j*result.getWidth()][0] = img.getVal(i,j);
                }
        }
        return result;

}
Image<PixRGB<float> > makeG(const Image<float>& img)
{
        Image<PixRGB<float> > result(img.getDims(),NO_INIT);

        Image<PixRGB<float> >::iterator optr = result.beginw();

        //Iterate over  and set values R,0,0

for(int i=0;i<result.getWidth();i++)
        {
                for(int j=0;j<result.getHeight();j++)
                {      // printf("i:%d  j:%d",i,j);

                          optr->setGreen(img.getVal(i,j));

                          optr->setRed(0);

                          optr->setBlue(0);
                        ++optr;
                }
        }
        return result;

        }

Image<PixRGB<float> > makeB(const Image<float>& img)
{
        Image<PixRGB<float> > result(img.getDims(),NO_INIT);

        Image<PixRGB<float> >::iterator optr = result.beginw();

        //Iterate over  and set values R,0,0
        for(int i=0;i<result.getWidth();i++)
        {
                for(int j=0;j<result.getHeight();j++)
                {      // printf("i:%d  j:%d",i,j);

                          optr->setBlue(img.getVal(i,j));

                          optr->setGreen(0);

                          optr->setRed(0);

                        optr=optr + 3 ;
                }
        }
        return result;

        if(Row%2==0 && j%2==0)
        {
          //Definiton of terms which will be used in kernel computation
                if(j<2) j_b = j_b + 2;
                if(Row<2) Row_b = Row_b + 2;
              //The code seems to work without taking care of the boundary condition ... is it safe to avoid it..will save some on computation..!!
                //if(j>(w-2)) j_m = j_m -2;
              //if(Row>(h-2)) Row_m = Row_m -2;

                A =         getVal[ j_b-1 + ((Row_b-1) * Width) ]+
                         getVal[ j+1 + ((Row_b-1) * Width) ]+
                         getVal[ j_b-1 + ((Row+1) * Width) ]+
                               getVal[ j+1 + ((Row+1) * Width) ];

                B =     getVal[ j + ((Row+1) * Width) ] +
                        getVal[ j + ((Row_b-1) * Width) ];

                C=      getVal[ j + ((Row+2) * Width) ]+
                        getVal[ j + ((Row_b-2) * Width) ];

                D=      getVal[ j_b-1 + ((Row) * Width) ]+
                        getVal[ j+1 + ((Row) * Width) ];

                E=      getVal[ j_b-2 + ((Row) * Width) ]+
                               getVal[ j+2 + ((Row) * Width) ];

                 F=      getVal[ j + ((Row) * Width) ];
                dptr[Row*Width+j].p[0]=  ((5*F) -(1 * (A + C)) + (4 * B) + ((1/2) * E))/8 ;
                dptr[Row*Width+j].p[1]=   F;
                dptr[Row*Width+j].p[2]=  ((5*F) -(1 * (A + E)) + (4 * D) + ((1/2) * C))/8 ;
        }
        if(Row%2==0 && j%2==1)
        {
          //Definiton of terms which will be used in kernel computation
                if(j<1) j_b = j_b + 2;
                if(Row<1) Row_b = Row_b + 2;
                A =         getVal[ j_b-1 + ((Row_b-1) * Width) ]+
                         getVal[ j+1 + ((Row_b-1) * Width) ]+
                         getVal[ j_b-1 + ((Row+1) * Width) ]+
                               getVal[ j+1 + ((Row+1) * Width) ];

                B =     getVal[ j + ((Row+1) * Width) ] +
                        getVal[ j + ((Row_b-1) * Width) ];

                C=      getVal[ j + ((Row+2) * Width) ]+
                        getVal[ j + ((Row_b-2) * Width) ];

                D=      getVal[ j_b-1 + ((Row) * Width) ]+
                        getVal[ j+1 + ((Row) * Width) ];

                E=      getVal[ j_b-2 + ((Row) * Width) ]+
                               getVal[ j+2 + ((Row) * Width) ];

                 F=      getVal[ j + ((Row) * Width) ];
                dptr[Row*Width+j].p[0]=  ((6*F) -((3/2) * (E + C)) + (2 * A)) /8 ;
                dptr[Row*Width+j].p[1]=  ((4*F) -(1 * (C + E)) + (2 * (D+B)))/8;
                dptr[Row*Width+j].p[2]=  F ;
        }
        if(Row%2==1 && j%2==0)
        {
           //Definiton of terms which will be used in kernel computation
                if(j<1) j_b = j_b + 2;
                if(Row<1) Row_b = Row_b + 2;
                A =         getVal[ j_b-1 + ((Row_b-1) * Width) ]+
                         getVal[ j+1 + ((Row_b-1) * Width) ]+
                         getVal[ j_b-1 + ((Row+1) * Width) ]+
                               getVal[ j+1 + ((Row+1) * Width) ];

                B =     getVal[ j + ((Row+1) * Width) ] +
                        getVal[ j + ((Row_b-1) * Width) ];

                C=      getVal[ j + ((Row+2) * Width) ]+
                        getVal[ j + ((Row_b-2) * Width) ];

                D=      getVal[ j_b-1 + ((Row) * Width) ]+
                        getVal[ j+1 + ((Row) * Width) ];

                E=      getVal[ j_b-2 + ((Row) * Width) ]+
                               getVal[ j+2 + ((Row) * Width) ];

                 F=      getVal[ j + ((Row) * Width) ];
                dptr[Row*Width+j].p[0]=  F ;
                dptr[Row*Width+j].p[1]=  ((4*F) -(1 * (C + E)) + (2 * (D+B)))/8;
                dptr[Row*Width+j].p[2]=  ((6*F) -((3/2) * (E + C)) + (2 * A)) /8 ;

        }
        if(Row%2==1 && j%2==1)
        {
          //Definiton of terms which will be used in kernel computation
                if(j<1) j_b = j_b + 2;
                if(Row<1) Row_b = Row_b + 2;
                A =         getVal[ j_b-1 + ((Row_b-1) * Width) ]+
                         getVal[ j+1 + ((Row_b-1) * Width) ]+
                         getVal[ j_b-1 + ((Row+1) * Width) ]+
                               getVal[ j+1 + ((Row+1) * Width) ];

                B =     getVal[ j + ((Row+1) * Width) ] +
                        getVal[ j + ((Row_b-1) * Width) ];

                C=      getVal[ j + ((Row+2) * Width) ]+
                        getVal[ j + ((Row_b-2) * Width) ];

                D=      getVal[ j_b-1 + ((Row) * Width) ]+
                        getVal[ j+1 + ((Row) * Width) ];

                E=      getVal[ j_b-2 + ((Row) * Width) ]+
                               getVal[ j+2 + ((Row) * Width) ];

                 F=      getVal[ j + ((Row) * Width) ];
                dptr[Row*Width+j].p[0]=  ((5*F) -(1 * (A + E)) + (4 * D) + ((1/2) * C))/8 ;
                dptr[Row*Width+j].p[1]=   F;
                dptr[Row*Width+j].p[2]=  ((5*F) -(1 * (A + C)) + (4 * B) + ((1/2) * E))/8;
        }

}*/
Image<PixRGB<float> > cpu_debayer_MHC(const Image<float>& img)
{
        //Initialize the image
        Image<PixRGB<float> > result(img.getDims(),NO_INIT);
        Image<PixRGB<float> >::iterator optr = result.beginw();
        Image<PixRGB<float> >::iterator end = result.endw();

        // printf("Width:%d Height:%d",result.getWidth(),result.getHeight());
        //Assign value to each pixel in the result image
        //Perhaps this is not a great idea to use this iterator
          int A,B,C,D,E,F;
        int i,i_m,i_b;
        int j,j_m,j_b;
        i=0;
        j=0;
        while(optr!=end)
        {
//                        if(i%2==0 &&  j%2==0)
        i_m = i;
        i_b = i;
        j_m = j;
        j_b = j;

//                        {
//                          optr->setRed(img.getVal(j,i+1));
if(i%2==0 && j%2==0)
        {
          //Definiton of terms which will be used in kernel computation
                if(i>=(result.getHeight()-2)) i_m= result.getHeight();
                if(j>=(result.getWidth()-2))  j_m= result.getWidth();
                if(j<2) j_b = 0;
                if(i<2) i_b = 0;
              //The code seems to work without taking care of the boundary condition ... is it safe to avoid it..will save some on computation..!!
                //if(j>(w-2)) j_m = j_m -2;
              //if(i>(h-2)) i_m = i_m -2;

                A =         img.getVal( j_b-1, i_b-1 )+
                         img.getVal( j_m+1 , i_b-1)+
                         img.getVal( j_b-1 , i_m+1)+
                               img.getVal( j_m+1 ,i_m+1 );

                B =     img.getVal( j, i_m+1 ) +
                        img.getVal( j, i_b-1) ;

                C=      img.getVal( j,i_m+2 )+
                        img.getVal( j, i_b-2 );

                D=      img.getVal( j_b-1,i )+
                        img.getVal( j_m+1 ,i );

                E=      img.getVal( j_b-2 ,i )+
                               img.getVal( j_m+2,i );

                 F=      img.getVal( j ,i );
                optr->setRed(  ((5*F) -(1 * (A + C)) + (4 * B) + ((1/2) * E))/8);
                optr->setGreen(F);
                optr->setBlue(((5*F) -(1 * (A + E)) + (4 * D) + ((1/2) * C))/8) ;
                        ++optr;

                        j++;

                        if(j==result.getWidth())
                        {
                                i++;
                                j=0;
                        }
        }
        else
        {
                       if(i%2==0 && j%2==1)
        {
          //Definiton of terms which will be used in kernel computation
                if(j<2) j_b = 0;
                if(i<2) i_b = 0;
                if(i>=(result.getHeight()-2)) i_m=result.getWidth();
                if(j>=(result.getWidth()-2)) j_m= result.getWidth();

                A =         img.getVal( j_b-1,i_b-1 )+
                         img.getVal( j_m+1,i_b-1 )+
                         img.getVal( j_b-1,i_m+1 )+
                               img.getVal( j_m+1,i_m+1 );

                B =     img.getVal( j,i_m+1 ) +
                        img.getVal( j,i_b-1 );

                C=      img.getVal( j,i_m+2 )+
                        img.getVal( j,i_b-2);

                D=      img.getVal( j_b-1,i )+
                        img.getVal( j_m+1 ,i );

                E=      img.getVal( j_b-2,i )+
                               img.getVal( j_m+2,i );

                 F=      img.getVal( j,i );
                optr->setRed(((6*F) -((3/2) * (E + C)) + (2 * A)) /8) ;
                optr->setGreen(((4*F) -(1 * (C + E)) + (2 * (D+B)))/8);
                optr->setBlue(F) ;
                        ++optr;

                        j++;

                        if(j==result.getWidth())
                        {
                                i++;
                                j=0;
                        }
        }
        else
        {
                       if(i%2==1 && j%2==0)
        {
           //Definiton of terms which will be used in kernel computation
                if(j<2) j_b = 0;
                if(i<2) i_b = 0;
                if(i>=(result.getHeight()-2)) i_m= result.getHeight();
                if(j>=(result.getWidth()-2)) j_m= result.getWidth();
                A =         img.getVal( j_b-1,i_b-1)+
                         img.getVal( j_m+1,i_b-1 )+
                         img.getVal( j_b-1,i_m+1 )+
                               img.getVal( j_m+1,i_m+1 );

                B =     img.getVal( j,i_m+1 ) +
                        img.getVal(j,i_b-1 );

                C=      img.getVal( j,i_m+2 )+
                        img.getVal( j,i_b-2 );

                D=      img.getVal( j_b-1,i )+
                        img.getVal( j_m+1,i );

                E=      img.getVal( j_b-2,i )+
                               img.getVal( j_m+2,i );

                 F=      img.getVal( j,i );
                optr->setRed(F) ;
                optr->setGreen(((4*F) -(1 * (C + E)) + (2 * (D+B)))/8);
                optr->setBlue(((6*F) -((3/2) * (E + C)) + (2 * A)) /8) ;
                        ++optr;

                        j++;

                        if(j==result.getWidth())
                        {
                                i++;
                                j=0;
                        }

        }
        else
        {
                       if(i%2==1 && j%2==1)
        {
          //Definiton of terms which will be used in kernel computation
                if(j<2) j_b = 0;
                if(i<2) i_b = 0;
                if(i>=(result.getHeight()-2)) i_m= result.getHeight();
                if(j>=(result.getWidth()-2)) j_m= result.getWidth();
                A =         img.getVal( j_b-1,i_b-1 )+
                         img.getVal( j_m+1,i_b-1 )+
                         img.getVal( j_b-1,i_m+1 )+
                               img.getVal( j_m+1,i_m+1 );

                B =     img.getVal( j,i_m+1 ) +
                        img.getVal( j,i_b-1 );

                C=      img.getVal( j,i_m+2 )+
                        img.getVal( j,i_b-2 );

                D=      img.getVal( j_b-1,i )+
                        img.getVal( j_m+1,i );

                E=      img.getVal( j_b-2,i )+
                               img.getVal( j_m+2,i );

                 F=      img.getVal( j,i );
                optr->setRed(((5*F) -(1 * (A + E)) + (4 * D) + ((1/2) * C))/8) ;
                optr->setGreen(F);
                optr->setBlue(((5*F) -(1 * (A + C)) + (4 * B) + ((1/2) * E))/8);
                        ++optr;

                        j++;

                        if(j==result.getWidth())
                        {
                                i++;
                                j=0;
                        }
        }//if loop ends here
        }// 1 st else
        }// 2 nd else
            }// 3 rd else



        }//while loop ends here


return result;
}//function defination ends here


Image<PixRGB<float> > cpu_debayer(const Image<float>& img)
{
        //Initialize the image
        Image<PixRGB<float> > result(img.getDims(),NO_INIT);
        Image<PixRGB<float> >::iterator optr = result.beginw();
        Image<PixRGB<float> >::iterator end = result.endw();

        // printf("Width:%d Height:%d",result.getWidth(),result.getHeight());
        //Assign value to each pixel in the result image
        //Perhaps this is not a great idea to use this iterator
        int i=0;
        int j=0;
        while(optr!=end)
        {
                        if(i%2==0 &&  j%2==0)
                        {
                          optr->setRed(img.getVal(j,i+1));

                          optr->setGreen(img.getVal(j,i));

                          optr->setBlue(img.getVal(j+1,i));
                        ++optr;

                        j++;

                        if(j==result.getWidth())
                        {
                                i++;
                                j=0;
                        }

                        }
                        if(i%2==1 &&  j%2==0)
                        {
                          optr->setRed(img.getVal(j,i));

                          optr->setGreen(img.getVal(j,i-1));

                          optr->setBlue(img.getVal(j+1,i-1));
                        ++optr;
                                j++;


                        if(j==result.getWidth())
                        {
                                i++;
                                j=0;
                        }

                        }
                        if(i%2==0 &&  j%2==1)
                        {
                          optr->setRed(img.getVal(j-1,i));

                          optr->setGreen(img.getVal(j,i+1));

                          optr->setBlue(img.getVal(j,i));
                        ++optr;                j++;

                        if(j==result.getWidth())
                        {
                                i++;
                                j=0;
                        }

                        }
                        if(i%2==1 &&  j%2==1)
                        {
                          optr->setRed(img.getVal(j-1,i));

                          optr->setGreen(img.getVal(j,i));

                          optr->setBlue(img.getVal(j,i-1));
                        ++optr;                j++;

                        if(j==result.getWidth())
                        {
                                i++;
                                j=0;
                        }

                        }//if ends

        }
        //Return the result
        return result;
}

void bayer_test(char **argv)
{

  Image<float> i = Raster::ReadFloat(argv[1]);

  int deviceNum = 1 ;
  CudaDevices::displayProperties(deviceNum);
 // int counter;
  CudaImage<PixRGB<float> > res_cuda;
  CudaImage<float> f;
  CudaImage<float> res_cuda_r_only;
  CudaImage<float> res_cuda_g_only;
  CudaImage<float> res_cuda_b_only;
  Image<PixRGB<float> > res_cuda_r;
  Image<PixRGB<float> > res_cuda_g;
  Image<PixRGB<float> > res_cuda_b;
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
//  for(counter=0;counter<10;counter++)
  {
                       f         =         CudaImage<float>(i,GLOBAL_DEVICE_MEMORY,deviceNum);

                res_cuda        = cuda_1_debayer(f);
  }
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime,start,stop);
  printf("Elasped Time GPU in ms for 1 iteration : %f\n",elapsedTime);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  //Get color components
  cudaGetComponents(res_cuda,res_cuda_r_only,res_cuda_g_only,res_cuda_b_only);

  //Make R,0,0 from r only
  Image<float> zero(res_cuda.getDims(),NO_INIT);

  res_cuda_r = makeRGB(res_cuda_r_only.exportToImage(),zero,zero);
  res_cuda_g = makeRGB(zero,res_cuda_g_only.exportToImage(),zero);
  res_cuda_b = makeRGB(zero,zero,res_cuda_b_only.exportToImage());





 // Image<PixRGB<float> > res_copy_cuda_r = res_cuda_r.exportToImage();

  Raster::WriteRGB(res_cuda_r,"test_gpu_r.ppm");

// Image<PixRGB<float> > res_copy_cuda_g = res_cuda_g.exportToImage();

  Raster::WriteRGB(res_cuda_g,"test_gpu_g.ppm");
 // Image<PixRGB<float> > res_copy_cuda_b = res_cuda_b.exportToImage();

  Raster::WriteRGB(res_cuda_b,"test_gpu_b.ppm");
//##########################################################
  Image<PixRGB<float> > res_copy_cuda = res_cuda.exportToImage();

  Raster::WriteRGB(res_copy_cuda,"test_gpu.ppm");
  /*
  struct timeval cstart,cend;
  long mtime,seconds,useconds;
 // int counter2;
  gettimeofday(&cstart,NULL);
  Image<PixRGB<float> > res_cpu;
//for(counter2=0;counter2<10;counter2++)
  res_cpu = cpu_debayer_MHC(i);

  gettimeofday(&cend,NULL);
  seconds = cend.tv_sec - cstart.tv_sec;
  useconds= cend.tv_usec - cstart.tv_usec;
  mtime = ((seconds)*1000 + useconds/1000);
  printf("Elapsed Time CPU in ms for 1 iteration : %ld\n",mtime);


  Raster::WriteRGB(res_cpu,"test_cpu.ppm");

  Image<float> res_cpu_r_only;
  Image<float> res_cpu_g_only;
  Image<float> res_cpu_b_only;

  Image<PixRGB<float> > res_cpu_r;
  Image<PixRGB<float> > res_cpu_g;
  Image<PixRGB<float> > res_cpu_b;

  //Make R,0,0 from r only
  Image<float> zero1(res_cpu.getDims(),NO_INIT);

  getComponents(res_cpu,res_cpu_r_only,res_cpu_g_only,res_cpu_b_only);
  res_cpu_r = makeRGB(res_cpu_r_only,zero,zero);
  res_cpu_g = makeRGB(zero,res_cpu_g_only,zero);
  res_cpu_b = makeRGB(zero,zero,res_cpu_b_only);

  Raster::WriteRGB(res_cpu_r,"test_cpu_r.ppm");
  Raster::WriteRGB(res_cpu_g,"test_cpu_g.ppm");
  Raster::WriteRGB(res_cpu_b,"test_cpu_b.ppm");




  testDiff(res_cpu_r_only,res_cuda_r_only.exportToImage());
  testDiff(res_cpu_g_only,res_cuda_g_only.exportToImage());
  testDiff(res_cpu_b_only,res_cuda_b_only.exportToImage());
  */
}


/*
void unit_test(int argc, char **argv)
{

  if (argc != 2) LFATAL("USAGE: %s <input.pgm>", argv[0]);

  setFpuRoundingMode(FPU_ROUND_NEAR);
  int cudaDeviceNum = 0;
  CudaDevices::displayProperties(cudaDeviceNum);

  //CUT_DEVICE_INIT(cudaDeviceNum);

  LINFO("Reading: %s", argv[1]);
  Image<PixRGB<float> > img = Raster::ReadRGB(argv[1]);



  // Compare normal implementation versus CUDA implementation
  // Original Implementation
  Image<float> normalLum = luminanceNTSC(img);
  // CUDA Implementation
  // Copy image to CUDA
  CudaImage<PixRGB<float> > cimg = CudaImage<PixRGB<float> >(img,GLOBAL_DEVICE_MEMORY,cudaDeviceNum);
  // Run CUDA Implementation and shove it back onto the host
  CudaImage<float> cLum = cudaLuminanceNTSC(cimg);
  Image<float> hcLum = cLum.exportToImage();
  testDiff(normalLum,hcLum);
  // Compare results
  // Test low pass 5 cuda filter against standard
  testDiff(cudaLowPass5Dec(cLum,true,true).exportToImage(), lowPass5yDecY(lowPass5xDecX(normalLum)));
  // Test low pass 9 filter against standard
  testDiff(cudaLowPass9(cLum,true,true).exportToImage(), lowPass9(normalLum,true,true));
  // Test the Gaussian Pyramid building
  CudaImageSet<float> cis = cudaBuildPyrGaussian(cLum,0,9,5);
  ImageSet<float> is = buildPyrGaussian(normalLum,0,9,5);
  for(uint i=0;i<is.size();i++)
  {
    Image<float> ctmp= cis[i].exportToImage();
    testDiff(ctmp,is[i]);
  }
}
*/
void toytest()
{
  int dev = 0;
  CudaDevices::displayProperties(dev);
  Image<float> img = Image<float>(10,10,NO_INIT);
  img.setVal(0,0,10.0F); img.setVal(0,1,20.0F); img.setVal(0,2,30.0F); img.setVal(0,3,40.0F); img.setVal(0,4,50.0F);
  img.setVal(0,5,20.0F); img.setVal(0,6,30.0F); img.setVal(0,7,40.0F); img.setVal(0,8,50.0F); img.setVal(0,9,60.0F);
  img.setVal(1,0,30.0F); img.setVal(1,1,40.0F); img.setVal(1,2,50.0F); img.setVal(1,3,60.0F); img.setVal(1,4,70.0F);
  img.setVal(0,5,40.0F); img.setVal(1,6,50.0F); img.setVal(1,7,60.0F); img.setVal(1,8,70.0F); img.setVal(1,9,80.0F);
  img.setVal(2,0,50.0F); img.setVal(2,1,60.0F); img.setVal(2,2,70.0F); img.setVal(2,3,80.0F); img.setVal(2,4,90.0F);
  img.setVal(2,5,60.0F); img.setVal(2,6,70.0F); img.setVal(2,7,80.0F); img.setVal(2,8,90.0F); img.setVal(2,9,100.F);
  img.setVal(3,0,70.0F); img.setVal(3,1,80.0F); img.setVal(3,2,90.0F); img.setVal(3,3,100.F); img.setVal(3,4,110.F);
  img.setVal(3,5,80.0F); img.setVal(3,6,90.0F); img.setVal(3,7,100.F); img.setVal(3,8,110.F); img.setVal(3,9,120.F);
  img.setVal(4,0,90.0F); img.setVal(4,1,100.F); img.setVal(4,2,110.F); img.setVal(4,3,120.F); img.setVal(4,4,130.F);
  img.setVal(4,5,80.0F); img.setVal(4,6,90.0F); img.setVal(4,7,100.F); img.setVal(4,8,110.F); img.setVal(4,9,120.F);
  img.setVal(5,0,70.0F); img.setVal(5,1,80.0F); img.setVal(5,2,90.0F); img.setVal(5,3,100.F); img.setVal(5,4,110.F);
  img.setVal(5,5,60.0F); img.setVal(5,6,70.0F); img.setVal(5,7,80.0F); img.setVal(5,8,90.0F); img.setVal(5,9,100.F);
  img.setVal(6,0,50.0F); img.setVal(6,1,60.0F); img.setVal(6,2,70.0F); img.setVal(6,3,80.0F); img.setVal(6,4,90.0F);
  img.setVal(6,5,40.0F); img.setVal(6,6,50.0F); img.setVal(6,7,60.0F); img.setVal(6,8,70.0F); img.setVal(6,9,80.0F);
  img.setVal(7,0,30.0F); img.setVal(7,1,40.0F); img.setVal(7,2,50.0F); img.setVal(7,3,60.0F); img.setVal(7,4,70.0F);
  img.setVal(7,5,20.0F); img.setVal(7,6,30.0F); img.setVal(7,7,40.0F); img.setVal(7,8,50.0F); img.setVal(7,9,60.0F);
  img.setVal(8,0,10.0F); img.setVal(8,1,20.0F); img.setVal(8,2,30.0F); img.setVal(8,3,40.0F); img.setVal(8,4,50.0F);
  img.setVal(8,5,00.0F); img.setVal(8,6,10.0F); img.setVal(8,7,20.0F); img.setVal(8,8,30.0F); img.setVal(8,9,40.0F);
  img.setVal(9,0,00.0F); img.setVal(9,1,00.0F); img.setVal(9,2,10.0F); img.setVal(9,3,20.0F); img.setVal(9,4,30.0F);
  img.setVal(9,5,00.0F); img.setVal(9,6,00.0F); img.setVal(9,7,00.0F); img.setVal(9,8,10.0F); img.setVal(9,9,20.0F);


  //Image<float> cres = cudaLowPass5yDec(img).exportToImage();
  //Image<float> normres = lowPass5yDecY(img);
  //testDiff(normres,cres);
  //printImages(normres,cres);
}


int main(int argc, char **argv)
{
 // unit_test(argc,argv);
  bayer_test(argv);
  //toytest();
}
