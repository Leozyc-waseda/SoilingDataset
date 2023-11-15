/*! @file Qt/GrabQtMainForm.ui.h main window for test-GrabQt */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt/GrabQtMainForm.ui.h $
// $Id: GrabQtMainForm.ui.h 10993 2009-03-06 06:05:33Z itti $

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

#include "Image/Image.H"
#include "Image/Pixels.H"

void GrabQtMainForm::init( int argc, const char **argv )
{
  grabbing = false;
  gbc.reset( new FrameGrabberConfigurator( manager ) );
  manager.addSubComponent( gbc );

  // parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false)
    exit( 1 );

  // use configuration dialog
  bool dorun = false;
  mmc.init( manager, &dorun );
  mmc.show();

  // post-config configs (!!!)
  gb = gbc->getFrameGrabber();
  if( gb.get() == NULL )
    {
      QMessageBox::warning( this, "Warning",
                            "You need to select a valid frame grabber for this program to be useful.",
                            QMessageBox::Ok, QMessageBox::NoButton,
                            QMessageBox::NoButton );
    }
}

void GrabQtMainForm::handlePauseButton()
{
  display = new QDialog( this );
  display->setFixedSize( gb->readRGB().getWidth(),
                         gb->readRGB().getHeight() );
  display->show();
  startTimer( 33 );
}

void GrabQtMainForm::handlePanSlider( int pos )
{
  QMessageBox::warning( this, "Warning",
                        "This has not been implemented yet!!",
                        QMessageBox::Ok, QMessageBox::NoButton,
                        QMessageBox::NoButton );
}

void GrabQtMainForm::handleTiltSlider( int pos )
{
  QMessageBox::warning( this, "Warning",
                        "This has not been implemented yet!!",
                        QMessageBox::Ok, QMessageBox::NoButton,
                        QMessageBox::NoButton );
}

void GrabQtMainForm::saveImage()
{
  QString file = QFileDialog::getSaveFileName( QString::null,
                                               "Images (*.pgm *.ppm)",
                                               this, "OpenImageDialog",
                                               "Choose Image" );
  if( !file.isEmpty() )
    Raster::WriteRGB( gb->readRGB(), file.latin1() );
  else
    QMessageBox::critical( this, "Error",
                           "Error: No image file specified!",
                           QMessageBox::Ok, QMessageBox::NoButton,
                           QMessageBox::NoButton );
}

void GrabQtMainForm::timerEvent( QTimerEvent *qte )
{
  grabImage();
}

void GrabQtMainForm::grabImage()
{
  QPixmap qpixm = convertToQPixmap( gb->readRGB() );
  display->setPaletteBackgroundPixmap( qpixm );
}
