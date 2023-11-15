/*! @file Qt/QtTestForm.ui.h main window for test-Qt */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt/QtTestForm.ui.h $
// $Id: QtTestForm.ui.h 4711 2005-06-27 23:53:44Z zhanshi $

/****************************************************************************
** ui.h extension file, included from the uic-generated form implementation.
**
** If you wish to add, delete or rename slots use Qt Designer which will
** update this file, preserving your code. Create an init() slot in place of
** a constructor, and a destroy() slot in place of a destructor.
*****************************************************************************/

#include <qmessagebox.h>

void QtTestForm::showAboutQt()
{
    QMessageBox::aboutQt( this, "iLab on Qt" );
}


void QtTestForm::showAboutiLab()
{
    QDialog iLabBox( this, "iLabBox", 1 );
    iLabBox.setCaption( "USC iLab -- Visit us at http://iLab.usc.edu/" );
    iLabBox.setPaletteBackgroundPixmap( *(iLabLogo->pixmap()) );
    iLabBox.setFixedSize( iLabLogo->pixmap()->width(), iLabLogo->pixmap()->height() );
    iLabBox.exec();
}
