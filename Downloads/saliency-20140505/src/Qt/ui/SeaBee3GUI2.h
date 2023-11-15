/****************************************************************************
** Form interface generated from reading ui file 'Qt/SeaBee3GUI2.ui'
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef SEABEE3MAINDISPLAYFORM_H
#define SEABEE3MAINDISPLAYFORM_H

#include <qvariant.h>
#include <qpixmap.h>
#include <qmainwindow.h>
#include <list>
#include <IceUtil/Mutex.h>
#include <queue>
#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/PixelsTypes.H"
#include "GUI/SimpleMeter.H"
#include "Util/MathFunctions.H"
#include "Robots/SeaBeeIII/MapperI.H"
#include "Image/CutPaste.H"
#include "Raster/PngParser.H"
#include "Raster/GenericFrame.H"
#include "Image/Point2D.H"

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class QAction;
class QActionGroup;
class QToolBar;
class QPopupMenu;
class ImageCanvas;
class QLineEdit;
class QGroupBox;
class QLabel;
class QCheckBox;
class QPushButton;
class QTabWidget;
class QWidget;
class QFrame;
class SeaBee3GUIIce;
class BeeStemData;

class SeaBee3MainDisplayForm : public QMainWindow
{
    Q_OBJECT

public:
    SeaBee3MainDisplayForm( QWidget* parent = 0, const char* name = 0, WFlags fl = WType_TopLevel );
    ~SeaBee3MainDisplayForm();

    QLineEdit* desired_heading_field_2_3;
    QLineEdit* desired_heading_field_2_2;
    QGroupBox* groupBox2_3_3_3;
    QLineEdit* desired_speed_field_3_3_3;
    QLineEdit* desired_depth_field_3_3_3;
    QLineEdit* desired_heading_field_3_3_3;
    QLabel* textLabel2_3_2_2_3_5_2_3_4_2_2_2;
    QLabel* textLabel2_3_2_2_3_5_2_3_4_2_2_2_2_2;
    QLabel* textLabel2_3_2_2_3_5_2_3_4_2_2;
    QLabel* textLabel2_3_2_2_3_5_2_3_4_2_2_2_2;
    QLabel* textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_2;
    QLabel* textLabel2_3_2_2_3_5_2_3_4_2;
    QLabel* textLabel2_2_2_3_2_2_3_2_2_3_3_2;
    QLabel* textLabel2_2_2_3_2_2_2_2_2_2_3_3;
    QLabel* textLabel2_2_2_3_2_2_2_2_2_2_3_3_2;
    QLabel* textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2;
    QLabel* textLabel2_3_2_2_3_5_2_3_4_2_2_2_2_2_2;
    QLabel* textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_2_2;
    QLineEdit* itsFwdRetinaMsgField;
    QLineEdit* itsBeeStemMsgField;
    QLineEdit* desired_heading_field_2_4_2_3_3_2_2_2;
    QLineEdit* desired_heading_field_2_4_2_3_3_2_2_2_2_2;
    QLineEdit* itsVisionMsgField;
    QLineEdit* itsDwnRetinaMsgField;
    QLabel* textLabel2_2_2_3_2_2_3_2_2_3_3;
    QGroupBox* groupBox10_2;
    QGroupBox* groupBox2_3_3_2_2;
    QLineEdit* desired_speed_field_3_3_2_2;
    QLineEdit* desired_depth_field_3_3_2_2;
    QLineEdit* desired_heading_field_3_3_2_2;
    QLabel* textLabel2_3_2_2_3_5_2_3_3;
    QLabel* textLabel2_3_2_2_3_5_2_3_2_2;
    QLabel* textLabel2_2_2_3_2_2_3_2_2_3_2_2;
    QLabel* textLabel2_2_2_3_2_2_2_2_2_2_3_2_2;
    QLabel* textLabel2_2_2_3_2_2_2_2_2_2_3_2_2_3;
    QLabel* textLabel2_2_2_3_2_2_3_2_2_3_2_2_3;
    QLineEdit* desired_heading_field_2_4_2_3_2_2_3;
    QLineEdit* desired_depth_field_2_2_2_3_2_2_3;
    QLineEdit* desired_speed_field_2_2_2_3_2_2_3;
    QLabel* textLabel2_3_2_2_3_5_2_3_3_3;
    QLabel* textLabel2_3_2_2_3_5_2_3_2_2_3;
    QLineEdit* desired_heading_field_2_4_2_3_2_2;
    QLineEdit* desired_depth_field_2_2_2_3_2_2;
    QLineEdit* desired_speed_field_2_2_2_3_2_2;
    ImageCanvas* EPressureCanvas_2;
    ImageCanvas* itsDepthPIDImageDisplay;
    QGroupBox* groupBox2_3_3;
    QLineEdit* desired_speed_field_3_3;
    QLineEdit* desired_depth_field_3_3;
    QLineEdit* desired_heading_field_3_3;
    QCheckBox* checkBox3_4_2_2_3;
    QPushButton* pushButton1;
    QLabel* textLabel2_2_2_3_2_2_2_2_2_2_3_4;
    QLabel* textLabel2_2_2_3_2_2_2_2_2_2_3;
    ImageCanvas* itsStrafeAxisImageDisplay_2;
    ImageCanvas* itsDepthAxisImageDisplay;
    QLabel* textLabel2_2_2_3_2_2_3_2_2_3;
    ImageCanvas* itsHeadingAxisImageDisplay;
    QLabel* textLabel2_3_2_2_3_5_2_3;
    ImageCanvas* itsStrafeAxisImageDisplay;
    QLineEdit* desired_speed_field_2_2_2_3_2_3_2;
    QGroupBox* groupBox2_3_3_2;
    QLineEdit* desired_speed_field_3_3_2;
    QLineEdit* desired_depth_field_3_3_2;
    QLineEdit* desired_heading_field_3_3_2;
    QCheckBox* checkBox3_4_2_2_3_2;
    QLabel* textLabel2_3_2_2_3_5_2_3_2;
    QLineEdit* desired_speed_field_2_2_2_3_2;
    QLabel* textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_3;
    QLineEdit* desired_depth_field_2_2_2_3_2;
    QLineEdit* desired_heading_field_2_4_2_3_2;
    QLabel* textLabel2_2_2_3_2_2_3_2_2_3_2;
    QLabel* textLabel2_2_2_3_2_2_2_2_2_2_3_2;
    QLabel* textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_3_2;
    QLineEdit* itsHeadingOutputField;
    QLineEdit* itsDepthOutputField;
    QGroupBox* groupBox1;
    QLabel* textLabel2_2_2_3_2_2_2_2_2_2_3_2_3;
    ImageCanvas* EPressureCanvas_3;
    QGroupBox* groupBox9_3;
    QCheckBox* checkBox3_4;
    QCheckBox* checkBox3_2_2;
    QCheckBox* checkBox3_3_2;
    QLineEdit* itsExternalPressureField;
    ImageCanvas* ItsDepthImageDisplay;
    QGroupBox* groupBox6;
    QCheckBox* checkBox3;
    QCheckBox* checkBox3_2;
    QCheckBox* checkBox3_3;
    ImageCanvas* itsCompassImageDisplay;
    QLineEdit* itsCompassHeadingField;
    QLineEdit* itsKillSwitchField;
    QGroupBox* groupBox10;
    QLineEdit* itsInternalPressureField;
    ImageCanvas* itsPressureImageDisplay;
    QTabWidget* tabWidget3;
    QWidget* tab;
    QGroupBox* groupBox15_2;
    QFrame* frame4;
    ImageCanvas* itsDwnImgDisplay;
    ImageCanvas* itsFwdImgDisplay;
    QCheckBox* itsPipeThreshCheck;
    QCheckBox* itsHoughVisionCheck;
    QCheckBox* itsFwdContourThreshCheck;
    QCheckBox* itsBuoyThreshCheck;
    ImageCanvas* itsFwdVisionDisplay;
    QCheckBox* itsSaliencyVisionCheck;
    QCheckBox* itsDwnContourVisionCheck;
    ImageCanvas* itsDwnVisionDisplay;
    QWidget* TabPage;

public slots:
    virtual void init( ModelManager * mgr );
    virtual void registerCommunicator( nub::soft_ref<SeaBee3GUIIce> c );
    virtual void setFwdImage( Image<PixRGB<byte> > & img );
    virtual void setDwnImage( Image<PixRGB<byte> > & img );
    virtual void setFwdVisionImage( Image<PixRGB<byte> > & img );
    virtual void setDwnVisionImage( Image<PixRGB<byte> > & img );
    virtual void setCompassImage( Image<PixRGB<byte> > & compassImage );
    virtual void setDepthImage( Image<PixRGB<byte> > & depthImage );
    virtual void setPressureImage( Image<PixRGB<byte> > & pressureImage );
    virtual void setDepthPIDImage( Image<PixRGB<byte> > & depthPIDImage );
    virtual void setAxesImages( Image<PixRGB<byte> > & heading, Image<PixRGB<byte> > & depth, Image<PixRGB<byte> > & strafe );
    virtual void setFwdRetinaMsgField( char f );
    virtual void setDwnRetinaMsgField( char f );
    virtual void setBeeStemMsgField( char f );
    virtual void setVisionMsgField( char f );
    virtual void setBeeStemData( BeeStemData & d );
    virtual void updateBuoySegmentCheck( bool state );
    virtual void updateSalientPointCheck( bool state );

protected:
    ModelManager *itsMgr;
    nub::soft_ref<SeaBee3GUIIce> GUIComm;


protected slots:
    virtual void languageChange();

private:
    QPixmap image0;

};

#endif // SEABEE3MAINDISPLAYFORM_H
