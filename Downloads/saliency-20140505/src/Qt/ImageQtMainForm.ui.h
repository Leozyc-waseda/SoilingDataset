/*! @file Qt/ImageQtMainForm.ui.h main window for test-ImageQt */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt/ImageQtMainForm.ui.h $
// $Id: ImageQtMainForm.ui.h 6191 2006-02-01 23:56:12Z rjpeters $

/****************************************************************************
** ui.h extension file, included from the uic-generated form implementation.
**
** If you wish to add, delete or rename slots use Qt Designer which will
** update this file, preserving your code. Create an init() slot in place of
** a constructor, and a destroy() slot in place of a destructor.
*****************************************************************************/

void ImageQtMainForm::setImageFile()
{
    QString file = QFileDialog::getOpenFileName( QString::null,
                                                 "Images (*.pgm *.ppm)",
                                                 this, "OpenImageDialog",
                                                 "Choose Image" );
    if( !file.isEmpty() )
        ImageFileLineEdit->setText( file );
    loadImage();
}


void ImageQtMainForm::displayImage()
{
    QPixmap qpixm = convertToQPixmap( img );
    if( FullSizeBox->isChecked() )
    {
        QDialog imgDialog( this, "ImageDisplay", 1 );
        imgDialog.setCaption( "Full-size Display" );
        imgDialog.setPaletteBackgroundPixmap( qpixm );
        imgDialog.setFixedSize( img.getWidth(), img.getHeight() );
        imgDialog.exec();
    }
    else
    {
        ImagePixmapLabel->setPixmap( qpixm );
    }
}


void ImageQtMainForm::loadImage()
{
    QString file = ImageFileLineEdit->text();
    if( !file.isEmpty() )
        img = Raster::ReadRGB( file );
    else
        QMessageBox::critical( this, "Error",
                               "Error: No image file specified!",
                               QMessageBox::Ok, QMessageBox::NoButton,
                               QMessageBox::NoButton );
}
