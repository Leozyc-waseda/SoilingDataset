/****************************************************************************
** Form interface generated from reading ui file 'Qt/ImageQtMainForm.ui'
**
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef IMAGEQTMAINFORM_H
#define IMAGEQTMAINFORM_H

#include <qvariant.h>
#include <qdialog.h>
#include "Image/Image.H"
#include "Image/Pixels.H"

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class QPushButton;
class QLineEdit;
class QLabel;
class QCheckBox;

class ImageQtMainForm : public QDialog
{
    Q_OBJECT

public:
    ImageQtMainForm( QWidget* parent = 0, const char* name = 0, bool modal = FALSE, WFlags fl = 0 );
    ~ImageQtMainForm();

    QPushButton* ChooseButton;
    QPushButton* ImageLoadButton;
    QLineEdit* ImageFileLineEdit;
    QLabel* ImageFileTextLabel;
    QCheckBox* FullSizeBox;
    QPushButton* DisplayButton;
    QLabel* ImagePixmapLabel;

public slots:
    virtual void setImageFile();
    virtual void displayImage();
    virtual void loadImage();

protected:
    Image< PixRGB<byte> > img;


protected slots:
    virtual void languageChange();

};

#endif // IMAGEQTMAINFORM_H
