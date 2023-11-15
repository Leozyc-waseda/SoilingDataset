

#include "BeoSub/ColorSegmenter.H"


ColorSegmenter::ColorSegmenter(){


   debugmode = false;
  setup = false;
  color.resize(4,0.0F);
  std.resize(4,0.0F);
  norm.resize(4,0.0F);
  adapt.resize(4,0.0F);
  lowerBound.resize(4,0.0F);
  upperBound.resize(4,0.0F);
  candidate_color = "none";
  Hz = 0.0F;
  fps = 0.0F;
  res = 0.0F;

  int wi = 320/4;
  int hi = 240/4;


  width = 320;
  height = 240;
  segmenter = new segmentImageTrackMC<float,unsigned int, 4> (width*height);

  segmenter->SITsetFrame(&wi,&hi);

  segmenter->SITsetCircleColor(0,255,0);
  segmenter->SITsetBoxColor(255,255,0,0,255,255);
  segmenter->SITsetUseSmoothing(false,10);


  segmenter->SITtoggleCandidateBandPass(false);
  segmenter->SITtoggleColorAdaptation(false);
}

ColorSegmenter::~ColorSegmenter(){
}


void ColorSegmenter::setupColorSegmenter(const char* inputColor, bool debug){
 debugmode = debug;
 //for now just allow it orange
 //candidate_color = inputColor;
 //inputColor = "Orange";
 // canidate_color = inputColor;

 //if(!strcmp(candidate_color, "Orange")){
    setupOrange();
    // }
    // else{
    // printf("Cannot setup decoder without a color to decode! exiting...\n");
    // return;
    // }

  segmenter->SITsetTrackColor(&color,&std,&norm,&adapt,&upperBound,&lowerBound);

  //if(debugmode){
    wini.reset(new XWindow(Dims(width, height), 0, 0, "test-input window"));
    wini->setPosition(0, 0);
    wino.reset(new XWindow(Dims(width, height), 0, 0, "test-output window"));
    wino->setPosition(width+10, 0);
    //}


  setup = true;
}

void ColorSegmenter::setupOrange() {

  colorConf.openFile("colortrack.conf", true);

  PixRGB<float> P1(colorConf.getItemValueF("ORANGE_R"),
                   colorConf.getItemValueF("ORANGE_G"),
                   colorConf.getItemValueF("ORANGE_B"));

  PixH2SV1<float> ORANGE(P1);

  color[0] = ORANGE.H1(); color[1] = ORANGE.H2(); color[2] = ORANGE.S(); color[3] = ORANGE.V();

  //! +/- tollerance value on mean for track
  std[0] = colorConf.getItemValueF("ORNG_std0");
  std[1] = colorConf.getItemValueF("ORNG_std1");
  std[2] = colorConf.getItemValueF("ORNG_std2");
  std[3] = colorConf.getItemValueF("ORNG_std3");

  //! normalizer over color values (highest value possible)
  norm[0] = colorConf.getItemValueF("ORNG_norm0");
  norm[1] = colorConf.getItemValueF("ORNG_norm1");
  norm[2] = colorConf.getItemValueF("ORNG_norm2");
  norm[3] = colorConf.getItemValueF("ORNG_norm3");

  //! how many standard deviations out to adapt, higher means less bias
  adapt[0] = colorConf.getItemValueF("ORNG_adapt0");
  adapt[1] = colorConf.getItemValueF("ORNG_adapt1");
  adapt[2] = colorConf.getItemValueF("ORNG_adapt2");
  adapt[3] = colorConf.getItemValueF("ORNG_adapt3");

  //! highest value for color adaptation possible (hard boundry)
  upperBound[0] = color[0] + colorConf.getItemValueF("ORNG_up0");
  upperBound[1] = color[1] + colorConf.getItemValueF("ORNG_up1");
  upperBound[2] = color[2] + colorConf.getItemValueF("ORNG_up2");
  upperBound[3] = color[3] + colorConf.getItemValueF("ORNG_up3");

  //! lowest value for color adaptation possible (hard boundry)
  lowerBound[0] = color[0] - colorConf.getItemValueF("ORNG_lb0");
  lowerBound[1] = color[1] - colorConf.getItemValueF("ORNG_lb1");
  lowerBound[2] = color[2] - colorConf.getItemValueF("ORNG_lb2");
  lowerBound[3] = color[3] - colorConf.getItemValueF("ORNG_lb3");
}

    /*
//true test with frame 3498
float ColorSegmenter::oldisolateOrange(Image< PixRGB<byte> > &inputImage,  Image<byte> &outputImage)
{

  readConfig colorConf;
  colorConf.openFile("colortrack.conf", false);

  int orangeCount = 0;

  Image<PixRGB<byte> >::iterator iptr = inputImage.beginw();
  Image<byte>::iterator          optr = outputImage.beginw();
  Image<PixRGB<byte> >::iterator stop = inputImage.endw();

  float tR = colorConf.getItemValueF("ORANGE_stdR");//70.0;
  float tG = colorConf.getItemValueF("ORANGE_stdG");//200.0;
  float tB = colorConf.getItemValueF("ORANGE_stdB");//128.0;

  float R = colorConf.getItemValueF("ORANGE_R");//255;
  float G = colorConf.getItemValueF("ORANGE_G");//198;
  float B = colorConf.getItemValueF("ORANGE_B");//0;


        //for average
        float totalHue = 0.0;
        int numTotal = 0;

        //orange as a hue
        float pure_orange_hue =  60*(((G/255)-(B/255))/((R/255)-(B/255)))+0;
        float orange_hue = 60*(((tB/255)-(tR/255))/((tG/255) - (tR/255)))+120;
        //orange saturation (of HSL)
        float orange_sat = ((200.0/255.0)-(70/255.0))/(2.0-(270.0/255.0));//using tR,tG,tB, R,B,G gives '1'
        std::cout<<"orange hue is: "<<orange_hue<<std::endl;
        std::cout<<"orange saturation(purity) is: "<<orange_sat<<std::endl;
        std::cout<<"orange HSV saturation is: "<<(1.0-70.0/200.0)<<std::endl;
//  LINFO("orange values (RGB):(std RGB): %f, %f, %f: %f, %f, %f", R, G, B, tR, tG, tB);
  while (iptr != stop)
    {
        float hue = 0.0;
        float s = 0.0; //saturation
      float avgR = (*iptr).red();
      float avgG = (*iptr).green();
      float avgB = (*iptr).blue();
        float r = avgR/255;
        float g = avgG/255;
        float b = avgB/255;




        //do conversion to HSV to find the hue
        float max = 0;
        float min = 1;
        //find max
        if(r > max) { max = r;}
        if(g > max) { max = g;}
        if(b > max) { max = b;}
        //find min
        if(r < min){min = r;}
        if(g < min){min = g;}
        if(b < min){min = b;}

        //do conversion to find hue
        if(max == min) {hue = 0.0;}
        else if(max == r && g >= b) {hue = 60.0*((g-b)/(max - min)) + 0.0;}
        else if(max == r && g < b) {hue = 60.0*((g-b)/(max - min)) + 360.0;}
        else if(max == g) {hue =  60.0*((b-r)/(max-min))+120.0;}
        else if(max == b) {hue = 60.0*((r-g)/(max-min))+240.0;}


        //for average calculation
        totalHue += hue;
        numTotal++;

        //find saturation
        if(max){s = max;}
        if(max != 0){s = 1 - min/max;}
        //std::cout<<" "<<hue;
        if(hue == orange_hue)//result:get spects here and there
        {
          //(*optr) = (byte)255;  // orange
          //orangeCount++;
        }
        if(hue == pure_orange_hue)//result:nothing
        {
          //(*optr) = (byte)255;  // orange
          //orangeCount++;
        }
        //to reason these numbers 145 is about the value of orange hue
        //pretty good but with spects, "s != 1" gets rid of specs, but also takes out some of the pipe
        //value of 120 to 145 seems best
        //using 130 as min makes it less accurate
        //using a higher max does not seem to make a difference
        //probably because of the colors involved here
        if(!(120<hue && hue<146) &&
        s != 1)
        {
        //std::cout<<" "<<s;
          (*optr) = (byte)255;  // orange
         orangeCount++;
        }

      //float avg = (avgR+avgG+avgB)/3.0;
      //float sigma = pow(avgR - avg, 2.) + pow(avgG - avg, 2.) + pow(avgB - avg, 2.);
      //float stdDev = sqrt( (1./3.) * sigma );

        //result: pretty good but is confused by highlights
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

        std::cout<<"average hue was "<<totalHue/numTotal<<std::endl;
  return float(orangeCount)/float( (inputImage.getHeight() * inputImage.getWidth()));
}


    */



//true test with frame 3498
float ColorSegmenter::isolateOrange(Image< PixRGB<byte> > &inputImage,  Image<byte> &outputImage)
{
  Image< PixRGB<byte> > tempImage(inputImage);
  Image<PixH2SV2<float> > h2svImage(tempImage);

  readConfig colorConf;
  colorConf.openFile("colortrack.conf", false);

  int orangeCount = 0;

  Image<PixRGB<byte> >::iterator iptr = inputImage.beginw();
  Image<byte>::iterator          optr = outputImage.beginw();
  Image<PixRGB<byte> >::iterator stop = inputImage.endw();

  float tR = colorConf.getItemValueF("ORANGE_stdR");//70.0;
  float tG = colorConf.getItemValueF("ORANGE_stdG");//200.0;
  float tB = colorConf.getItemValueF("ORANGE_stdB");//128.0;

  float R = colorConf.getItemValueF("ORANGE_R");//255;
  float G = colorConf.getItemValueF("ORANGE_G");//198;
  float B = colorConf.getItemValueF("ORANGE_B");//0;


  //for average
  float totalHue = 0.0;
  int numTotal = 0;

  //orange as a hue
  float pure_orange_hue =  60*(((G/255)-(B/255))/((R/255)-(B/255)))+0;
  float orange_hue = 60*(((tB/255)-(tR/255))/((tG/255) - (tR/255)))+120;
  //orange saturation (of HSL)
  float orange_sat = ((200.0/255.0)-(70/255.0))/(2.0-(270.0/255.0));//using tR,tG,tB, R,B,G gives '1'

  std::cout<<"orange hue is: "<<orange_hue<<std::endl;
  std::cout<<"orange saturation(purity) is: "<<orange_sat<<std::endl;
  std::cout<<"orange HSV saturation is: "<<(1.0-70.0/200.0)<<std::endl;
  //  LINFO("orange values (RGB):(std RGB): %f, %f, %f: %f, %f, %f", R, G, B, tR, tG, tB);

  while (iptr != stop)
    {
      float hue = 0.0;
      float s = 0.0; //saturation
      float avgR = (*iptr).red();
      float avgG = (*iptr).green();
      float avgB = (*iptr).blue();
      float r = avgR/255;
      float g = avgG/255;
      float b = avgB/255;




      //do conversion to HSV to find the hue
      float max = 0;
      float min = 1;
      //find max
      if(r > max) { max = r;}
      if(g > max) { max = g;}
      if(b > max) { max = b;}
      //find min
      if(r < min){min = r;}
      if(g < min){min = g;}
      if(b < min){min = b;}

      //do conversion to find hue
      if(max == min) {hue = 0.0;}
      else if(max == r && g >= b) {hue = 60.0*((g-b)/(max - min)) + 0.0;}
      else if(max == r && g < b) {hue = 60.0*((g-b)/(max - min)) + 360.0;}
      else if(max == g) {hue =  60.0*((b-r)/(max-min))+120.0;}
      else if(max == b) {hue = 60.0*((r-g)/(max-min))+240.0;}


      //for average calculation
      totalHue += hue;
      numTotal++;

      //find saturation
      if(max){s = max;}
      if(max != 0){s = 1 - min/max;}
      //std::cout<<" "<<hue;
      if(hue == orange_hue)//result:get spects here and there
        {
          //(*optr) = (byte)255;  // orange
          //orangeCount++;
        }
      if(hue == pure_orange_hue)//result:nothing
        {
          //(*optr) = (byte)255;  // orange
          //orangeCount++;
        }
        //to reason these numbers 145 is about the value of orange hue
        //pretty good but with spects, "s != 1" gets rid of specs, but also takes out some of the pipe
        //value of 120 to 145 seems best
        //using 130 as min makes it less accurate
        //using a higher max does not seem to make a difference
        //probably because of the colors involved here
      if(!(120<hue && hue<146) &&
         s != 1)
        {
        //std::cout<<" "<<s;
          (*optr) = (byte)255;  // orange
          orangeCount++;
        }

      //float avg = (avgR+avgG+avgB)/3.0;
      //float sigma = pow(avgR - avg, 2.) + pow(avgG - avg, 2.) + pow(avgB - avg, 2.);
      //float stdDev = sqrt( (1./3.) * sigma );

        //result: pretty good but is confused by highlights
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

      //display image to compare to what we get with the color segmenter

      Image<PixRGB<byte> > Aux;
      Aux.resize(100,450,true);

      /******************************************************************/
      // SEGMENT IMAGE ON EACH INPUT FRAME

      segmenter->SITtrackImageAny(h2svImage,&inputImage,&Aux,true);

      /* Retrieve and Draw all our output images */
      Image<byte> temp = quickInterpolate(segmenter->SITreturnCandidateImage(),4);
      //display for now for testing purposes
      //wini->drawImage(display);
      wino->drawImage(temp);

      std::cout<<"average hue was "<<totalHue/numTotal<<std::endl;
      return float(orangeCount)/float( (inputImage.getHeight() * inputImage.getWidth()));
}





