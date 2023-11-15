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
#include "Qt/SeaBee3GUIIce.H"

void SeaBee3MainDisplayForm::init( ModelManager *mgr )
{
  itsMgr = mgr;
}

void SeaBee3MainDisplayForm::registerCommunicator( nub::soft_ref<SeaBee3GUIIce> c )
{
  LINFO("Registering Communicator");
  GUIComm = c;
}

void SeaBee3MainDisplayForm::setFwdImage(Image<PixRGB<byte> > &img)
{
      itsFwdImgDisplay->setImage(img);
 }

void SeaBee3MainDisplayForm::setDwnImage(Image<PixRGB<byte> > &img)
{
      itsDwnImgDisplay->setImage(img);
 }

void SeaBee3MainDisplayForm::setFwdVisionImage(Image<PixRGB<byte> > &img)
{
      itsFwdVisionDisplay->setImage(img);
 }

void SeaBee3MainDisplayForm::setDwnVisionImage(Image<PixRGB<byte> > &img)
{
      itsDwnVisionDisplay->setImage(img);
 }

void SeaBee3MainDisplayForm::setCompassImage(Image<PixRGB<byte> > &compassImage)
{
      itsCompassImageDisplay->setImage(compassImage);
}

void SeaBee3MainDisplayForm::setDepthImage(Image<PixRGB<byte> > &depthImage)
{
     ItsDepthImageDisplay->setImage(depthImage);
}

void SeaBee3MainDisplayForm::setPressureImage(Image<PixRGB<byte> > &pressureImage)
{
     itsPressureImageDisplay->setImage(pressureImage);
}

void SeaBee3MainDisplayForm::setDepthPIDImage(Image<PixRGB<byte> > &depthPIDImage)
{
     itsDepthPIDImageDisplay->setImage(depthPIDImage);
}

void SeaBee3MainDisplayForm::setAxesImages(Image<PixRGB<byte> > &heading, Image<PixRGB<byte> > &depth, Image<PixRGB<byte> > &strafe)
{
     itsHeadingAxisImageDisplay->setImage(heading);
     itsDepthAxisImageDisplay->setImage(depth);
     itsStrafeAxisImageDisplay->setImage(strafe);
}

void SeaBee3MainDisplayForm::setFwdRetinaMsgField(char f)
{
  itsFwdRetinaMsgField->setText(toStr<int>(f));
}

void SeaBee3MainDisplayForm::setDwnRetinaMsgField(char f)
{
  itsDwnRetinaMsgField->setText(toStr<int>(f));
}

void SeaBee3MainDisplayForm::setBeeStemMsgField(char f)
{
  itsBeeStemMsgField->setText(toStr<int>(f));
}

void SeaBee3MainDisplayForm::setVisionMsgField(char f)
{
  itsVisionMsgField->setText(toStr<int>(f));
}

void SeaBee3MainDisplayForm::setBeeStemData(BeeStemData &d)
{
  itsCompassHeadingField->setText(toStr<int>(d.heading));
  itsInternalPressureField->setText(toStr<int>(d.internalPressure));
  itsExternalPressureField->setText(toStr<int>(d.externalPressure));
  itsHeadingOutputField->setText(toStr<int>(d.headingPIDOutput));
  itsDepthOutputField->setText(toStr<int>(d.depthPIDOutput));

  //TODO   itsKillSwitchField->setPaletteForegroundColor()
  itsKillSwitchField->setText((d.killSwitch == true ) ? "ON" : "OFF");
}



void SeaBee3MainDisplayForm::updateBuoySegmentCheck( bool state )
{
    GUIComm->setOrangeSegEnabled(state);
}

void SeaBee3MainDisplayForm::updateSalientPointCheck( bool state )
{
    GUIComm->setSalientPointsEnabled(state);
}
