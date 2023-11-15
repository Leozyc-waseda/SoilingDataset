#ifndef ISOLATE_COLOR
#define ISOLATE_COLOR

#include "Image/Image.H"
#include "Image/PixelsTypes.H"

// #include "BeoSub/HoughTransform.H"

// #define GREEN

// int isolateBlack(const Image< PixRGB<byte> > &inputImage,  Image<byte> &outputImage) {
//   int blackCount = 0;
//   for(int j = 0; j < inputImage.getHeight(); j++ ) {
//     for(int i = 0; i < inputImage.getWidth(); i++ )
//       {

//         /* float h1 = inputImage.getVal(i, j).H1();
//            float h2 = inputImage.getVal(i, j).H2();
//            float s = inputImage.getVal(i, j).S();
//            float v = inputImage.getVal(i, j).V();*/


//         float avgR = inputImage.getVal(i, j).red();
//         float avgG = inputImage.getVal(i, j).green();
//         float avgB = inputImage.getVal(i, j).blue();

//         float avg = (avgR+avgG+avgB)/3.0;
//         float sigma = pow(avgR - avg, 2.) + pow(avgG - avg, 2.) + pow(avgB - avg, 2.);
//         float stdDev = sqrt( (1./3.) * sigma );


//         if (avgG > avgR && avgG > avgB && stdDev > 15) {

//           if (outputImage.coordsOk(i, j)) {

//             blackCount++;
//             outputImage.setVal(i, j, 255);
//           }

//         }
//         else {
//           if(outputImage.coordsOk(i,j)){
//             outputImage.setVal(i, j, 0);

//           }

//         }

//       }
//   }
//   //return blackCount;
//   return (blackCount*100)/(inputImage.getHeight() * inputImage.getWidth());
// }

// int isolateGreen(const Image< PixRGB<byte> > &inputImage,  Image<byte> &outputImage) {
//   int blackCount = 0;
//   for(int j = 0; j < inputImage.getHeight(); j++ ) {
//     for(int i = 0; i < inputImage.getWidth(); i++ )
//       {

//         /* float h1 = inputImage.getVal(i, j).H1();
//            float h2 = inputImage.getVal(i, j).H2();
//            float s = inputImage.getVal(i, j).S();*/
//         float v = ((PixH2SV2 <float> )inputImage.getVal(i, j)).V();


//         /*          float avgR = inputImage.getVal(i, j).red();
//                     float avgG = inputImage.getVal(i, j).green();
//                     float avgB = inputImage.getVal(i, j).blue();

//                     float avg = (avgR+avgG+avgB)/3.0;
//                     float sigma = pow(avgR - avg, 2.) + pow(avgG - avg, 2.) + pow(avgB - avg, 2.);
//                     float stdDev = sqrt( (1./3.) * sigma );


//                     if (avgG > avgR && avgG > avgB && stdDev > 15) {
//         */
//         if (v < 0.3) {
//           if (outputImage.coordsOk(i, j)) {

//             blackCount++;
//             outputImage.setVal(i, j, 255);
//           }

//         }
//         else {
//           if(outputImage.coordsOk(i,j)){
//             outputImage.setVal(i, j, 0);

//           }

//         }

//       }
//   }
//   //return blackCount;
//   return (blackCount*100)/(inputImage.getHeight() * inputImage.getWidth());
// }

// int isolateWhite(const Image< PixRGB<byte> > &inputImage,  Image<byte> &outputImage, int& x, int& y) {
//   int whiteCount = 0;
//   x = 0; y = 0;
//   for(int j = 0; j < inputImage.getHeight(); j++ ) {
//     for(int i = 0; i < inputImage.getWidth(); i++ )
//       {



//         //float h1 = ((PixH2SV2 <float> )inputImage.getVal(i, j)).H1();
//         //float h2 = ((PixH2SV2 <float> )inputImage.getVal(i, j)).H2();
//         float s = ((PixH2SV2 <float> )inputImage.getVal(i, j)).S();
//         float v = ((PixH2SV2 <float> )inputImage.getVal(i, j)).V();

//         //printf("\tS: %f V: %f", s, v);
//         if (/*h1 > 0.25 && h1 < 0.733333  && h2 > 0.0 && h2 < 0.312500 &&*//* s > 0.020490 && */s < 0.30 && v > 0.85 /*&& v < 1.0*/ ) {
//           //        if (s < 0.07 && v > 0.48) {
//           //printf("FOUND");
//           if (outputImage.coordsOk(i,j)) {
//             //printf("OK");
//             outputImage.setVal(i, j, 0);
//             whiteCount++;
//             x+=i;
//             y+=j;

//           }
//         }

//         else if(outputImage.coordsOk(i,j)) {
//           outputImage.setVal(i, j, 255);
//         }
//         //  printf("H1: %f %f %f %f\t", h1, h2, s, v);
//         /*
//           float avgR = inputImage.getVal(i, j).red();
//           float avgG = inputImage.getVal(i, j).green();
//           float avgB = inputImage.getVal(i, j).blue();

//           float avg = (avgR+avgG+avgB)/3.0;
//           float sigma = pow(avgR - avg, 2.) + pow(avgG - avg, 2.) + pow(avgB - avg, 2.);
//           float stdDev = sqrt( (1./3.) * sigma );

//           if (stdDev < 10 && avg > 200 && outputImage.coordsOk(i,j)) {
//           outputImage.setVal(i, j, 0);
//           whiteCount++;

//           }
//           else if(outputImage.coordsOk(i,j)) {
//           outputImage.setVal(i, j, 255);
//           }*/
//       }
//   }

//   if(whiteCount) {
//     x /= whiteCount;
//     y /= whiteCount;
//   }
//   else {
//     x = -1;
//     y = -1;
//   }

//   return (whiteCount*100)/(inputImage.getHeight() * inputImage.getWidth());
// }


float isolateOrange(Image< PixRGB<byte> > &inputImage,  Image<byte> &outputImage)
{
  LINFO("happy");
  int orangeCount = 0;

  Image<PixRGB<byte> >::iterator iptr = inputImage.beginw();
  Image<byte>::iterator          optr = outputImage.beginw();
  Image<PixRGB<byte> >::iterator stop = inputImage.endw();

//   float tR = 150.0;
//   float tG = 200.0;
//   float tB = 128.0;

//   float R = 255;
//   float G = 198;
//   float B = 0;

  float tR = 150.0;
  float tG = 200.0;
  float tB = 128.0;

  float R = 255;
  float G = 198;
  float B = 0;


  while (iptr != stop)
    {
      float avgR = (*iptr).red();
      float avgG = (*iptr).green();
      float avgB = (*iptr).blue();

      //float avg = (avgR+avgG+avgB)/3.0;
      //float sigma = pow(avgR - avg, 2.) + pow(avgG - avg, 2.) + pow(avgB - avg, 2.);
      //float stdDev = sqrt( (1./3.) * sigma );

      if (avgR > R - tR && avgR < R + tR &&
          avgG > G - tG && avgG < G + tG &&
          avgB > B - tB && avgB < B + tB   )
        {
          (*optr) = (byte)255;  // orange
          orangeCount++;
        }
//       else
//         {
//           //if(outputImage.coordsOk(i,j)){
//           //(*optr) = (byte)0; //not orange
//           //}
//         }

      iptr++; optr++;
    }
  return float(orangeCount)/float( (inputImage.getHeight() * inputImage.getWidth()));
}



// int isolateRed(const Image< PixRGB<byte> > &inputImage,  Image<byte> &outputImage) {
//   int blackCount = 0;
//   for(int j = 0; j < inputImage.getHeight(); j++ ) {
//     for(int i = 0; i < inputImage.getWidth(); i++ )
//       {

//         /* float h1 = inputImage.getVal(i, j).H1();
//            float h2 = inputImage.getVal(i, j).H2();
//            float s = inputImage.getVal(i, j).S();
//            float v = inputImage.getVal(i, j).V();*/


//         float avgR = inputImage.getVal(i, j).red();
//         float avgG = inputImage.getVal(i, j).green();
//         float avgB = inputImage.getVal(i, j).blue();

//         float avg = (avgR+avgG+avgB)/3.0;
//         float sigma = pow(avgR - avg, 2.) + pow(avgG - avg, 2.) + pow(avgB - avg, 2.);
//         float stdDev = sqrt( (1./3.) * sigma );

//         if (avgR > avgG && avgR > avgB && stdDev > 15 && outputImage.coordsOk(i,j)) {
//           outputImage.setVal(i, j, 255);
//           blackCount++;

//         }
//         else {
//           if(outputImage.coordsOk(i,j)){
//             outputImage.setVal(i, j, 0);
//           }
//         }

//       }
//   }

//   return (blackCount*100)/(inputImage.getHeight() * inputImage.getWidth());
// }

// // int houghTransform(Image< PixRGB <byte> >  &image, std::vector<LineSegment2D> &lines, int houghThreshold = 25, bool copyOver = false) {
// //                 float sigma = .7;
// //                 float tlow = 0.2;
// //                 float thigh = .97;
// //                 unsigned char *edge;

// //                 char *dirfilename = NULL;



// //                 Image<byte> byteImage = luminance(image);


// //                 Image< PixRGB<byte> > houghImg = byteImage;

// //                 canny(byteImage.getArrayPtr(), byteImage.getHeight(), byteImage.getWidth(), sigma, tlow, thigh, &edge, dirfilename);
// //                 Image<unsigned char> edgeImage(edge, byteImage.getWidth(), byteImage.getHeight());
// //                 houghImg = edgeImage;

// //                         //perform the hough transform on the green isolated image
// //                 lines = houghTransform(edgeImage, PI/180, 1, houghThreshold, houghImg);

// //         if (copyOver) {
// //                 image = luminance(houghImg);
// //         }

// //                 return lines.size();

// // }

// std::vector<LineSegment2D> pruneIntersections(std::vector<LineSegment2D> &lines, double angle=-1) {
//   std::vector<LineSegment2D> lines2;
//   int size = lines.size();
//   double x, y;

//   bool bad[size];


//   for (int z = 0; z < size; z++) bad[z] = false;

//   if (size > 1) {

//     for (int i = 0; i < size; i++) {
//       for (int a = 0; a < size; a++) {
//         if (a == i) continue;


//         if (lines[i].intersects(lines[a], x, y) && (angle == -1 || lines[i].angleBetween(lines[a]) < angle)) {
//           //whenever two lines intersect, prune the shorter one. as long as neither of them have been flagged as intersecting before
//           if (!bad[a] && !bad[i]) {

//             int x = lines[i].length() > lines[a].length() ? a : i;
//             bad[x] = true;
//           }
//         }

//       }
//     }
//     /*
//       for (int i = 0; i < size - 1; i++) {
//       //printf("Line: %d %d, %d %d\n", lines[i].point1().i, lines[i].point1().j, lines[i].point2().i, lines[i].point2().j);

//       for (int a = i + 1; a < size; a++) {

//       if (lines[i].intersects(lines[a], x, y)) {
//       //printf("Intersects at: %f, %f\n", x, y);
//       bad[i] = true;
//       break;
//       }
//       else {
//       bad[i] = false;
//       }
//       }
//       }*/
//   }

//   for (int i = 0; i < size; i++) {
//     if (!bad[i]) {
//       lines2.push_back(lines[i]);
//     }
//   }


//   return lines2;
// }
#endif
