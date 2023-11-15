/****************************************************************************
 ** ui.h extension file, included from the uic-generated form implementation.
 **
 ** If you want to add, delete, or rename functions or slots, use
 ** Qt Designer to update this file, preserving your code.
 **
 ** You should not define a constructor or destructor in this file.
 ** Instead, write your code in functions called init() and destroy().
 ** These will automatically be called by the form's constructor and
 ** destructor.
 *****************************************************************************/
#include "Util/StringConversions.H"
#include "Image/DrawOps.H"
#include "Image/ColorOps.H"
#include "Ice/RobotBrainObjects.ice.H"
#include "Raster/Raster.H"

#include <string.h>
#include <qevent.h>
#include <iostream>
#include <fstream>
#include <string>

#define H2SV_TYPE PixH2SV2

void SeaBee3MainDisplayForm::pushFwdRectangle( ImageIceMod::QuadrilateralIce quad)
{
  itsDataMutex.lock();
  itsFwdRectangles.push_back(quad);
  itsDataMutex.unlock();
}

void SeaBee3MainDisplayForm::setSensorVotes( std::vector<ImageIceMod::SensorVote> votes )
{
  itsDataMutex.lock();
  itsSensorVotes = votes;
  itsDataMutex.unlock();
}

void SeaBee3MainDisplayForm::setCompositeHeading( int heading )
{
  itsDataMutex.lock();
  itsCompositeHeading = heading;
  itsDataMutex.unlock();
}


void SeaBee3MainDisplayForm::setCompositeDepth( int depth )
{
  itsDataMutex.lock();
  itsCompositeDepth = depth;
  itsDataMutex.unlock();
}

void SeaBee3MainDisplayForm::pushRectRatio(float r)
{
  itsDataMutex.lock();
  itsRectRatios.push_back(r);
  itsDataMutex.unlock();
}

void SeaBee3MainDisplayForm::pushRectAngle(float a)
{
  itsDataMutex.lock();
  itsRectAngles.push_back(a);
  itsDataMutex.unlock();
}

void SeaBee3MainDisplayForm::pushContourPoint(Point2D<int> p)
{
  itsDataMutex.lock();
  itsRectangles.push_back(p);
  itsDataMutex.unlock();
}

void SeaBee3MainDisplayForm::setSalientPoint(Point2D<float> p)
{
  itsDataMutex.lock();
  itsSalientPoint = p;
  itsDataMutex.unlock();
}

Image<PixRGB<byte> > SeaBee3MainDisplayForm::makeCompassImage(int heading)
{
  headingHist.push_front(heading);
  if(headingHist.size() > 10)
    headingHist.pop_back();

  int w =CompassCanvas->width();
  int h = CompassCanvas->height();

  Image<PixRGB<byte> > compassImage(w,h,ZEROS);

  drawCircle(compassImage,Point2D<int>(w/2,h/2), .9*std::min(w,h)/2, PixRGB<byte>(255,0,0));

  std::list<int>::reverse_iterator it = headingHist.rbegin();
  float opacity=0;
  float opacity_step =1/float(headingHist.size());
  for(;it != headingHist.rend(); ++it)
    {
      //LINFO("Opacity:%f, Step: %f", opacity, opacity_step);
      int x = (int)(.9*std::min(w,h)/2*cos((*it-90)*(M_PI/180))); //shift by 90 so that 0 is up
      int y = (int)(.9*std::min(w,h)/2*sin((*it-90)*(M_PI/180)));
      drawArrow(compassImage, Point2D<int>(w/2,h/2), Point2D<int>(w/2+x,h/2+y), PixRGB<byte>(0,255*opacity*.5,255*opacity*.5));
      opacity+=opacity_step;
    }

  int x = (int)(.9*std::min(w,h)/2*cos((*(headingHist.begin())-90)*(M_PI/180))); //shift by 90 so that 0 is up
  int y = (int)(.9*std::min(w,h)/2*sin((*(headingHist.begin())-90)*(M_PI/180)));
  drawArrow(compassImage, Point2D<int>(w/2,h/2), Point2D<int>(w/2+x,h/2+y), PixRGB<byte>(0,255,0));


  //  int dx = (int)(100.0*cos((itsDesiredHeading-90)*(M_PI/180))); //shift by 90 so that 0 is up
  //  int dy = (int)(100.0*sin((itsDesiredHeading-90)*(M_PI/180)));
  // drawArrow(itsSubImage, Point2D<int>(128,128), Point2D<int>(128+dx,128+dy), PixRGB<byte>(0,0,255));

  return compassImage;
}
void SeaBee3MainDisplayForm::Image1Click( int a, int b, int c )
{
  LINFO("a: %d b: %d c: %d", a, b, c);

}

void SeaBee3MainDisplayForm::init( ModelManager *mgr )
{

  initRandomNumbers();
  itsMgr = mgr;

  int minCurrent = 0;
  int maxCurrent = 100;
  currentMeter0= new SimpleMeter(ThrusterCurrentMeterCanvas0->width(), ThrusterCurrentMeterCanvas0->height(), minCurrent, maxCurrent, (maxCurrent-minCurrent)*.60+minCurrent, (maxCurrent-minCurrent)*.90+minCurrent);

  currentMeter1= new SimpleMeter(ThrusterCurrentMeterCanvas1->width(), ThrusterCurrentMeterCanvas1->height(), minCurrent, maxCurrent, (maxCurrent-minCurrent)*.60+minCurrent, (maxCurrent-minCurrent)*.90+minCurrent);

  currentMeter2= new SimpleMeter(ThrusterCurrentMeterCanvas2->width(), ThrusterCurrentMeterCanvas2->height(), minCurrent, maxCurrent, (maxCurrent-minCurrent)*.60+minCurrent, (maxCurrent-minCurrent)*.90+minCurrent);

  currentMeter3= new SimpleMeter(ThrusterCurrentMeterCanvas3->width(), ThrusterCurrentMeterCanvas3->height(), minCurrent, maxCurrent, (maxCurrent-minCurrent)*.60+minCurrent, (maxCurrent-minCurrent)*.90+minCurrent);

  currentMeter4= new SimpleMeter(ThrusterCurrentMeterCanvas4->width(), ThrusterCurrentMeterCanvas4->height(), minCurrent, maxCurrent, (maxCurrent-minCurrent)*.60+minCurrent, (maxCurrent-minCurrent)*.90+minCurrent);

  currentMeter5= new SimpleMeter(ThrusterCurrentMeterCanvas5->width(), ThrusterCurrentMeterCanvas5->height(), minCurrent, maxCurrent, (maxCurrent-minCurrent)*.60+minCurrent, (maxCurrent-minCurrent)*.90+minCurrent);

  currentMeter6= new SimpleMeter(ThrusterCurrentMeterCanvas6->width(), ThrusterCurrentMeterCanvas6->height(), minCurrent, maxCurrent, (maxCurrent-minCurrent)*.60+minCurrent, (maxCurrent-minCurrent)*.90+minCurrent);

  currentMeter7= new SimpleMeter(ThrusterCurrentMeterCanvas7->width(), ThrusterCurrentMeterCanvas7->height(), minCurrent, maxCurrent, (maxCurrent-minCurrent)*.60+minCurrent, (maxCurrent-minCurrent)*.90+minCurrent);

  Image<PixRGB<byte> > meter0Img = currentMeter0->render(0);
  ThrusterCurrentMeterCanvas0->setImage(meter0Img);

  Image<PixRGB<byte> > meter1Img = currentMeter1->render(0);
  ThrusterCurrentMeterCanvas1->setImage(meter1Img);

  Image<PixRGB<byte> > meter2Img = currentMeter2->render(0);
  ThrusterCurrentMeterCanvas2->setImage(meter2Img);

  Image<PixRGB<byte> > meter3Img = currentMeter3->render(0);
  ThrusterCurrentMeterCanvas3->setImage(meter3Img);

  Image<PixRGB<byte> > meter4Img = currentMeter4->render(0);
  ThrusterCurrentMeterCanvas4->setImage(meter4Img);

  Image<PixRGB<byte> > meter5Img = currentMeter5->render(0);
  ThrusterCurrentMeterCanvas5->setImage(meter5Img);

  Image<PixRGB<byte> > meter6Img = currentMeter6->render(0);
  ThrusterCurrentMeterCanvas6->setImage(meter6Img);

  Image<PixRGB<byte> > meter7Img = currentMeter7->render(0);
  ThrusterCurrentMeterCanvas7->setImage(meter7Img);

  itsCompositeHeading = 0;
  itsCurrentHeading = 0;
  itsCurrentDepth = 0;
}

void SeaBee3MainDisplayForm::setImage(Image<PixRGB<byte> > &img, std::string cameraID)
{
  if(cameraID=="FwdCamera")
    {
      ImageDisplay0->setImage(img);

      itsDataMutex.lock();

      img = luminance(img);

      // draw movement controller related images
      Image<PixRGB<byte> > movementImg = img;
      Image<PixRGB<byte> > headingImg = Image<PixRGB<byte> >(movementImg.getDims(), ZEROS);
      Image<PixRGB<byte> >bufferImg(img.getDims(), ZEROS);
      std::vector<PixRGB<byte> > sensorVoteColors;
      sensorVoteColors.push_back(PixRGB<byte>(255,0,0));
      sensorVoteColors.push_back(PixRGB<byte>(0,255,0));
      sensorVoteColors.push_back(PixRGB<byte>(0,0,255));
      sensorVoteColors.push_back(PixRGB<byte>(0,0,255));

      uint w = movementImg.getWidth();
      uint h = movementImg.getHeight();

      for(uint i = 0; i < itsSensorVotes.size(); i++)
        {
          ImageIceMod::SensorVote sv = itsSensorVotes[i];

          std::vector<Point2D<int> > pts;
          pts.push_back(Point2D<int>((w/2) - 20 , 0));
          pts.push_back(Point2D<int>((sv.heading.val - itsCurrentHeading)+ w/2,
                                     (sv.depth.val - itsCurrentDepth)+ h/2));
          pts.push_back(Point2D<int>((w/2) - 20 , h));
          pts.push_back(Point2D<int>((w/2) + 20 , h));
          pts.push_back(Point2D<int>((sv.heading.val - itsCurrentHeading)+ w/2,
                                     (sv.depth.val - itsCurrentDepth)+ h/2));
          pts.push_back(Point2D<int>((w/2) + 20 , 0));
          drawFilledPolygon(bufferImg,
                            pts,
                            (PixRGB<byte>)(sensorVoteColors.at(i%sensorVoteColors.size()) *
                                           (sv.heading.weight + sv.depth.weight)/2));
          headingImg += bufferImg;
        }

      headingImg = headingImg * 0.9;
      movementImg += headingImg;

      MovementDisplay0->setImage(movementImg);
      movementImg = img;

      Image<PixRGB<byte> > compHeadingImg = Image<PixRGB<byte> >(movementImg.getDims(), ZEROS);
      std::vector<Point2D<int> > pts;
      pts.push_back(Point2D<int>((w/2) - 20 , 0));
      pts.push_back(Point2D<int>((itsCompositeHeading - itsCurrentHeading)+ w/2,
                                 (itsCompositeDepth - itsCurrentDepth)+ h/2));
      pts.push_back(Point2D<int>((w/2) - 20 , h));
      pts.push_back(Point2D<int>((w/2) + 20 , h));
      pts.push_back(Point2D<int>((itsCompositeHeading - itsCurrentHeading)+ w/2,
                                 (itsCompositeDepth - itsCurrentDepth)+ h/2));
      pts.push_back(Point2D<int>((w/2) + 20 , 0));
      drawFilledPolygon(compHeadingImg,pts,PixRGB<byte>(0,255,0));

      compHeadingImg = compHeadingImg * 0.9;
      movementImg += compHeadingImg;

      MovementDisplay1->setImage(movementImg);

            // draw fwd rectangles
      for(uint i = 0; i < itsFwdRectangles.size(); i++)
      {
          ImageIceMod::QuadrilateralIce quad = itsFwdRectangles[i];

          drawLine(img,Point2D<int>(quad.tr.i,quad.tr.j),
                   Point2D<int>(quad.br.i,quad.br.j),
                   PixRGB<byte>(0,255,0),2);
          drawLine(img,Point2D<int>(quad.br.i,quad.br.j),
                   Point2D<int>(quad.bl.i,quad.bl.j),
                   PixRGB<byte>(0,255,0),2);
          drawLine(img,Point2D<int>(quad.bl.i,quad.bl.j),
                   Point2D<int>(quad.tl.i,quad.tl.j),
                   PixRGB<byte>(0,255,0),2);
          drawLine(img,Point2D<int>(quad.tl.i,quad.tl.j),
                   Point2D<int>(quad.tr.i,quad.tr.j),
                   PixRGB<byte>(0,255,0),2);

          char* str = new char[20];
          sprintf(str,"%d,%d",quad.center.i,quad.center.j);
          writeText(img,Point2D<int>(quad.center.i,quad.center.j),str);
          delete [] str;
      }

      itsFwdRectangles.clear();
      // draw salient point
      Point2D<int> drawSalientPoint((float)(img.getWidth())*itsSalientPoint.i,(float)(img.getHeight())*itsSalientPoint.j);
      drawCircle(img,drawSalientPoint,7,PixRGB<byte>(100,0,0),3);
      drawCross(img,drawSalientPoint,PixRGB<byte>(255,0,0),5);
      drawCircle(img,drawSalientPoint,7,PixRGB<byte>(255,0,0));
      itsDataMutex.unlock();

      ImageDisplay1->setImage(img);

    }
  else if(cameraID=="DwnCamera")
    {
      ImageDisplay2->setImage(img);

      itsDataMutex.lock();
      if(itsRectangles.size() > 0)
        {
          std::vector<Point2D<int> >::iterator iter;
          std::vector<Point2D<int> > temp;

          for( uint i = 0; i < itsRectangles.size(); i++ ) {
            temp.push_back(itsRectangles[i]);

            if((i % 4)== 3)
              {
                Point2D<int> fp = temp[0];

                for(uint j = 0; j < temp.size(); j++)
                  {
                    drawLine(img,temp[j],temp[(j + 1) % 4],PixRGB<byte>(0,255,0),3);
                  }

                if(itsRectRatios.size() > 0 && itsRectAngles.size() > 0)
                  {
                    char* str = new char[20];
                    sprintf(str,"%1.2f, %2.1f",itsRectRatios.front(),itsRectAngles.front());
                    writeText(img,fp,str);
                    delete [] str;

                    itsRectRatios.pop_front();
                    itsRectAngles.pop_front();
                  }
                temp.clear();
              }
          }

          itsRectangles.clear();
        }

      itsDataMutex.unlock();

      ImageDisplay3->setImage(img);
    }

}

//There must be a better way to do this... --Rand
void SeaBee3MainDisplayForm::ToggleCamera0()
{
  static bool toggle = true;
  toggle = !toggle;
  ToggleCamera("Camera0", toggle);
}
void SeaBee3MainDisplayForm::ToggleCamera1()
{
  static bool toggle = true;
  toggle = !toggle;
  ToggleCamera("Camera1", toggle);
}
void SeaBee3MainDisplayForm::ToggleCamera2()
{
  static bool toggle = true;
  toggle = !toggle;
  ToggleCamera("Camera2", toggle);
}
void SeaBee3MainDisplayForm::ToggleCamera3()
{
  static bool toggle = true;
  toggle = !toggle;
  ToggleCamera("Camera3", toggle);
}

void SeaBee3MainDisplayForm::ToggleCamera(std::string cameraID, bool active)
{
  LINFO("Sending Toggle Camera Message");
  GUIComm->toggleCamera(cameraID,active);
}


void SeaBee3MainDisplayForm::registerCommunicator( nub::soft_ref<SeaBee3GUICommunicator> c )
{
  LINFO("Registering Communicator");
  GUIComm = c;
}



void SeaBee3MainDisplayForm::setJSAxis( int axis, float val )
{
  /*   std::string dispTxt;
       char b[4];
       sprintf(b,"%2.0f",val*100);
       dispTxt += b;

       switch(axis)
       {
       case 0:
       field_js_XAxis->setText(dispTxt);
       break;
       case 1:
       field_js_YAxis->setText(dispTxt);
       break;
       }*/
}


void SeaBee3MainDisplayForm::setSensorValues( int heading, int pitch, int roll,
                                              int intPressure, int extPressure, int headingValue,
                                              int depthValue )
{

  Image<PixRGB<byte> > compassImage = makeCompassImage(heading);
  CompassCanvas->setImage(compassImage);

  /*intPressHist.push_back(intPressure);
    if(intPressHist.size() > 40) intPressHist.pop_front();
    std::vector<float> intPressVec(intPressHist.begin(), intPressHist.end());

    int imin=0;
    int imax=0;
    if(IntPressAuto->isChecked())
    {
    imin = *(std::min_element(intPressVec.begin(), intPressVec.end()));
    imax = *(std::max_element(intPressVec.begin(), intPressVec.end()));
    IntPressMin->setValue(imin);
    IntPressMax->setValue(imax);
    }
    else
    {
    imin = IntPressMin->value();
    imax = IntPressMax->value();
    }

    Image<PixRGB<byte> > intPressImage = linePlot(intPressVec,
    IPressureCanvas->width(),
    IPressureCanvas->height(),
    imin,
    imax);
    IPressureCanvas->setImage(intPressImage);


    extPressHist.push_back(extPressure);
    if(extPressHist.size() > 40) extPressHist.pop_front();
    std::vector<float> extPressVec(extPressHist.begin(), extPressHist.end());

    int emin=0;
    int emax=0;
    if(ExtPressAuto->isChecked())
    {
    emin = *(std::min_element(extPressVec.begin(), extPressVec.end()));
    emax = *(std::max_element(extPressVec.begin(), extPressVec.end()));
    ExtPressMin->setValue(emin);
    ExtPressMax->setValue(emax);
    }
    else
    {
    emin = ExtPressMin->value();
    emax = ExtPressMax->value();
    }

    Image<PixRGB<byte> > extPressImage = linePlot(extPressVec,
    EPressureCanvas->width(),
    EPressureCanvas->height(),
    emin,
    emax
    );
    EPressureCanvas->setImage(extPressImage);*/


  itsCurrentHeading = heading;

  field_heading->setText(toStr<int>(heading));
  field_int_press->setText(toStr<int>(intPressure));
  field_ext_press->setText(toStr<int>(extPressure));
  heading_output_field->setText(toStr<int>(headingValue));
  depth_output_field->setText(toStr<int>(depthValue));
}


void SeaBee3MainDisplayForm::updateDesiredHeading( )
{
  QString val = desired_heading_field->text();
  // check that val is a number and that it is not empty
  if(strspn(val,"1234567890") != strlen(val) || strlen(val) == 0) return;

  int headingVal;
  convertFromString(val,headingVal);
  GUIComm->updatePoseSettings(1,headingVal,-1,-1);
}




void SeaBee3MainDisplayForm::updateDesiredDepth()
{
  QString val = desired_depth_field->text();
  // check that val is a number and that it is not empty
  if(strspn(val,"1234567890") != strlen(val) || strlen(val) == 0) return;

  int depthVal;
  convertFromString(val,depthVal);
  GUIComm->updatePoseSettings(2,-1,depthVal,-1);
}


void SeaBee3MainDisplayForm::updateDesiredSpeed()
{
  QString val = desired_speed_field->text();
  // check that val is a number and that it is not empty
  if(strspn(val,"1234567890") != strlen(val) || strlen(val) == 0) return;

  int speedVal;
  convertFromString(val,speedVal);
  GUIComm->updatePoseSettings(3,-1,-1,speedVal);
}


void SeaBee3MainDisplayForm::updateHeadingPID()
{
  QString kVal = field_heading_k->text();
  QString pVal = field_heading_p->text();
  QString iVal = field_heading_i->text();
  QString dVal = field_heading_d->text();

  // check that val is a number and that it is not empty
  if(strspn(kVal,"1234567890.") != strlen(kVal) || strlen(kVal) == 0) return;
  if(strspn(pVal,"1234567890.") != strlen(pVal) || strlen(pVal) == 0) return;
  if(strspn(iVal,"1234567890.") != strlen(iVal) || strlen(iVal) == 0) return;
  if(strspn(dVal,"1234567890.") != strlen(dVal) || strlen(dVal) == 0) return;

  float k,p,i,d;
  convertFromString(kVal,k);
  convertFromString(pVal,p);
  convertFromString(iVal,i);
  convertFromString(dVal,d);
  GUIComm->updatePID(1,k,p,i,d);
}


void SeaBee3MainDisplayForm::updateDepthPID()
{
  QString kVal = field_depth_k->text();
  QString pVal = field_depth_p->text();
  QString iVal = field_depth_i->text();
  QString dVal = field_depth_d->text();

  // check that val is a number and that it is not empty
  if(strspn(kVal,"1234567890.") != strlen(kVal) || strlen(kVal) == 0) return;
  if(strspn(pVal,"1234567890.") != strlen(pVal) || strlen(pVal) == 0) return;
  if(strspn(iVal,"1234567890.") != strlen(iVal) || strlen(iVal) == 0) return;
  if(strspn(dVal,"1234567890.") != strlen(dVal) || strlen(dVal) == 0) return;

  float k,p,i,d;
  convertFromString(kVal,k);
  convertFromString(pVal,p);
  convertFromString(iVal,i);
  convertFromString(dVal,d);
  GUIComm->updatePID(0,k,p,i,d);
}




void SeaBee3MainDisplayForm::manualClicked()
{
  GUIComm->disablePID();
}


void SeaBee3MainDisplayForm::autoClicked()
{
  GUIComm->enablePID();
}

//For SeaBeeInjector Tab
void SeaBee3MainDisplayForm::sendInitDone(){
  GUIComm->SeaBeeInjectorMsg(1,0,0,0,0,0,0,0,0,0);
}
void SeaBee3MainDisplayForm::sendGateFound(){
  GUIComm->SeaBeeInjectorMsg(0,1,0,0,0,0,0,0,0,0);
}
void SeaBee3MainDisplayForm::sendGateDone(){
  GUIComm->SeaBeeInjectorMsg(0,0,1,0,0,0,0,0,0,0);
}
void SeaBee3MainDisplayForm::sendContourFoundFlare(){
  GUIComm->SeaBeeInjectorMsg(0,0,0,1,0,0,0,0,0,0);
}
void SeaBee3MainDisplayForm::sendFlareDone(){
  GUIComm->SeaBeeInjectorMsg(0,0,0,0,1,0,0,0,0,0);
}
void SeaBee3MainDisplayForm::sendContourFoundBarbwire(){
  GUIComm->SeaBeeInjectorMsg(0,0,0,0,0,1,0,0,0,0);
}
void SeaBee3MainDisplayForm::sendBarbwireDone(){
  GUIComm->SeaBeeInjectorMsg(0,0,0,0,0,0,1,0,0,0);
}
void SeaBee3MainDisplayForm::sendContourFoundBoxes(){
  GUIComm->SeaBeeInjectorMsg(0,0,0,0,0,0,0,1,0,0);
}
void SeaBee3MainDisplayForm::sendBombingRunDone(){
  GUIComm->SeaBeeInjectorMsg(0,0,0,0,0,0,0,0,1,0);
}
void SeaBee3MainDisplayForm::sendBriefcaseFound(){
  GUIComm->SeaBeeInjectorMsg(0,0,0,0,0,0,0,0,0,1);
}


/*
  void SeaBee3MainDisplayForm::drawPipe(int x, int y){
  DrawRandomCirc();
  }



  void SeaBee3MainDisplayForm::drawBuoy(int x, int y){

  }


  void SeaBee3MainDisplayForm::drawBin(int x, int y){

  }


  void SeaBee3MainDisplayForm::drawBarbedWire(int x, int y){

  }


  void SeaBee3MainDisplayForm::drawMachineGunNest(int x, int y){

  }


  void SeaBee3MainDisplayForm::drawOctagonSurface(int x, int y){

  }
*/

void SeaBee3MainDisplayForm::addPlatform(){
  ObjectList->insertItem("Platform");
  MapperI::MapObject mapObject = {Point2D<float>(), Point2D<float>(), RobotSimEvents::PLATFORM};
  mapObjects.push_back(mapObject);
}

void SeaBee3MainDisplayForm::addGate(){
  ObjectList->insertItem("Gate");
  MapperI::MapObject mapObject = {Point2D<float>(), Point2D<float>(), RobotSimEvents::GATE};
  mapObjects.push_back(mapObject);
}

void SeaBee3MainDisplayForm::addPipe(){
  ObjectList->insertItem("Pipe");
  MapperI::MapObject mapObject = {Point2D<float>(), Point2D<float>(), RobotSimEvents::PIPE};
  mapObjects.push_back(mapObject);
}

void SeaBee3MainDisplayForm::addFlare(){
  ObjectList->insertItem("Flare");
  MapperI::MapObject mapObject = {Point2D<float>(), Point2D<float>(),RobotSimEvents::FLARE};
  mapObjects.push_back(mapObject);
}

void SeaBee3MainDisplayForm::addBin(){
  ObjectList->insertItem("Bin");
  MapperI::MapObject mapObject = {Point2D<float>(), Point2D<float>(), RobotSimEvents::BIN};
  mapObjects.push_back(mapObject);
}


void SeaBee3MainDisplayForm::addBarbwire(){
  ObjectList->insertItem("Barbwire");
  MapperI::MapObject mapObject = {Point2D<float>(), Point2D<float>(), RobotSimEvents::BARBWIRE};
  mapObjects.push_back(mapObject);
}


void SeaBee3MainDisplayForm::addMachineGunNest(){
  ObjectList->insertItem("MachineGunNest");
  MapperI::MapObject mapObject = {Point2D<float>(),Point2D<float>(), RobotSimEvents::MACHINEGUN};
  mapObjects.push_back(mapObject);
}


void SeaBee3MainDisplayForm::addOctagonSurface(){
  ObjectList->insertItem("OctagonSurface");
  MapperI::MapObject mapObject = {Point2D<float>(),Point2D<float>(), RobotSimEvents::OCTAGON};
  mapObjects.push_back(mapObject);
}

void SeaBee3MainDisplayForm::selectObject(int index){
  selectedIndex = index;
  MapObjectX->setText(convertToString(mapObjects[index].pos.i));
  MapObjectY->setText(convertToString(mapObjects[index].pos.j));
}

void SeaBee3MainDisplayForm::refreshMapImage(){

  MapImage = Image<PixRGB<byte> >(MapCanvas->width(), MapCanvas->height(), ZEROS);
  PngParser *parser = NULL;
  for (unsigned int i=0; i<mapObjects.size(); i++){
    if (mapObjects[i].type == RobotSimEvents::PLATFORM){
      parser = new PngParser("platform.png");
    }
    else if (mapObjects[i].type == RobotSimEvents::GATE){
      parser = new PngParser("gate_final.png");
    }
    else if (mapObjects[i].type == RobotSimEvents::PIPE){
      parser = new PngParser("pipe_final.png");
    }
    else if (mapObjects[i].type == RobotSimEvents::FLARE){
      parser = new PngParser("flare_final.png");
    }
    else if (mapObjects[i].type == RobotSimEvents::BIN){
      parser = new PngParser("bin_horizontal.png");
    }
    else if (mapObjects[i].type == RobotSimEvents::BARBWIRE){
      parser = new PngParser("barbwire_final.png");
    }
    else if (mapObjects[i].type == RobotSimEvents::MACHINEGUN){
      parser = new PngParser("machinegun_final.png");
    }
    else if (mapObjects[i].type == RobotSimEvents::OCTAGON){
      parser = new PngParser("octagon_final.png");
    }
    Image<PixRGB<byte> > src = parser->getFrame().asRgb();
    inplacePaste(MapImage, src, Point2D<int>(mapObjects[i].pos.i, mapObjects[i].pos.j));
    delete parser;
  }
  MapCanvas->setImage(MapImage);
}

void SeaBee3MainDisplayForm::moveObject(){
  int x = 0;
  int y = 0;
  convertFromString(MapObjectX->text(), x);
  convertFromString(MapObjectY->text(), y);
  if ( !(x > MapCanvas->width()-50 || x < 0 || y > MapCanvas->height()-50 || y < 0) ){
    mapObjects[selectedIndex].pos = Point2D<float>(x,y);
    refreshMapImage();
  }
  // Draw image to canvas
  // Raster/PngParser.H
  //PngParser parser("pipe_final.png");
  // Raster/GenericFrame.H

  /* Put a dummy image up there for now
     MapImage = Image<PixRGB<byte> >(MapCanvas->width(), MapCanvas->height(), ZEROS);
     Image<PixRGB<byte> > src = parser.getFrame().asRgb();
     inplacePaste(MapImage, src, Point2D<int>(x,y));
     MapCanvas->setImage(MapImage);
  */
}

void SeaBee3MainDisplayForm::deleteObject(){
  mapObjects.erase(mapObjects.begin() + selectedIndex);
  ObjectList->removeItem(selectedIndex);
  refreshMapImage();
}

void SeaBee3MainDisplayForm::saveMap(){
  ofstream ostream;
  string filename = MapName->text();
  if (filename.length() > 0){
    ostream.open(filename.c_str());
    if (ostream.is_open()){
      MapName->setText("");
      for (unsigned int        i=0; i<mapObjects.size(); i++){
        ostream << mapObjects[i].type << " " << mapObjects[i].pos.i << " " << mapObjects[i].pos.j << endl;
      }
      ostream.close();
    }
  }
}

void SeaBee3MainDisplayForm::loadMap(){
  string input;
  //RobotSimEvents::SeaBeeObjectType type;
  int x, y, z;
  ifstream istream;
  string filename = MapName->text();
  if (filename.length() > 0){
    istream.open(filename.c_str());
    if (istream.is_open()){
      MapName->setText("");
      mapObjects.clear();
      ObjectList->clear();
      while (true){
        istream >> z >> x >> y;
        if (istream.eof()){
          break;
        }
        //cout << z << " " << x << " " << y << endl;
        MapperI::MapObject mapObject = {Point2D<float>(x,y),Point2D<float>(), (RobotSimEvents::SeaBeeObjectType)z};
        mapObjects.push_back(mapObject);
        if (mapObject.type == RobotSimEvents::PLATFORM){
          ObjectList->insertItem("Platform");
        }
        else if (mapObject.type == RobotSimEvents::GATE){
          ObjectList->insertItem("Gate");
        }
        else if (mapObject.type == RobotSimEvents::PIPE){
          ObjectList->insertItem("Pipe");
        }
        else if (mapObject.type == RobotSimEvents::FLARE){
          ObjectList->insertItem("Flare");
        }
        else if (mapObject.type == RobotSimEvents::BIN){
          ObjectList->insertItem("Bin");
        }
        else if (mapObject.type == RobotSimEvents::BARBWIRE){
          ObjectList->insertItem("Barbwire");
        }
        else if (mapObject.type == RobotSimEvents::MACHINEGUN){
          ObjectList->insertItem("Machinegun");
        }
        else if (mapObject.type == RobotSimEvents::OCTAGON){
          ObjectList->insertItem("Octagon");
        }
        else{
          cout << "Error: Object undefined " << mapObject.type << endl;
        }
      }
      refreshMapImage();
    }
    //cout << ObjectList->count() << " " << mapObjects.size();
  }
}




void SeaBee3MainDisplayForm::setThrusterMeters( int zero, int one, int two, int three, int four, int five, int six, int seven )
{


  Image<PixRGB<byte> > meter1Img = currentMeter1->render(abs(one));
  ThrusterCurrentMeterCanvas1->setImage(meter1Img);

  Image<PixRGB<byte> > meter2Img = currentMeter2->render(abs(two));
  ThrusterCurrentMeterCanvas2->setImage(meter2Img);

  Image<PixRGB<byte> > meter3Img = currentMeter3->render(abs(three));
  ThrusterCurrentMeterCanvas3->setImage(meter3Img);

  Image<PixRGB<byte> > meter4Img = currentMeter4->render(abs(four));
  ThrusterCurrentMeterCanvas4->setImage(meter4Img);

  Image<PixRGB<byte> > meter5Img = currentMeter5->render(abs(five));
  ThrusterCurrentMeterCanvas5->setImage(meter5Img);

  Image<PixRGB<byte> > meter6Img = currentMeter6->render(abs(six));
  ThrusterCurrentMeterCanvas6->setImage(meter6Img);

}

Image<PixRGB<byte> >   itsInput;
Image<PixH2SV2<float> > itsConvert;

void SeaBee3MainDisplayForm::loadColorPickerImg( )
{
    std::string filename = colorFileLoadText->text();

    if(filename != "")
    {
        LINFO("Reading %s",filename.c_str());
        itsInput    = Raster::ReadRGB(filename);
        Image<PixRGB<float> >  finput   = itsInput;

             itsConvert =  static_cast<Image<PixH2SV2<float> > >(finput);

             ColorPickerImg->setImage(itsInput);
   }
}

void SeaBee3MainDisplayForm::clickColorPickerImg(int x, int y, int but)
{
    LINFO("Clicked!");
}
