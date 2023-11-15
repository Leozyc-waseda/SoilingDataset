/****************************************************************************
** Form interface generated from reading ui file 'Qt/SeaBee3GUI.ui'
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
#include "Qt/SeaBee3GUICommunicator.H"
#include "GUI/SimpleMeter.H"
#include "Util/MathFunctions.H"
#include "Robots/SeaBeeIII/MapperI.H"
#include "Image/CutPaste.H"
#include "Raster/PngParser.H"
#include "Raster/GenericFrame.H"
#include "Image/Point2D.H"
#include "Ice/RobotSimEvents.ice.H"

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class QAction;
class QActionGroup;
class QToolBar;
class QPopupMenu;
class ImageCanvas;
class QGroupBox;
class QFrame;
class QLabel;
class QLineEdit;
class QCheckBox;
class QSpinBox;
class QButtonGroup;
class QRadioButton;
class QTabWidget;
class QWidget;
class QPushButton;
class QLCDNumber;
class QComboBox;
class SeaBee3GUICommunicator;

class SeaBee3MainDisplayForm : public QMainWindow
{
    Q_OBJECT

public:
    SeaBee3MainDisplayForm( QWidget* parent = 0, const char* name = 0, WFlags fl = WType_TopLevel );
    ~SeaBee3MainDisplayForm();

    QGroupBox* groupBox9_4;
    QFrame* frame3;
    ImageCanvas* ThrusterCurrentMeterCanvas4;
    ImageCanvas* ThrusterCurrentMeterCanvas5;
    ImageCanvas* ThrusterCurrentMeterCanvas2;
    ImageCanvas* ThrusterCurrentMeterCanvas3;
    ImageCanvas* ThrusterCurrentMeterCanvas6;
    QLabel* textLabel2;
    QLabel* textLabel2_2;
    QLabel* textLabel2_2_2;
    QLabel* textLabel2_2_3;
    QLabel* textLabel2_2_4;
    QLabel* textLabel2_2_5;
    QLabel* textLabel2_2_6;
    QLabel* textLabel2_2_7;
    ImageCanvas* ThrusterCurrentMeterCanvas7;
    QLabel* textLabel1_3;
    QLabel* textLabel1_2;
    QLabel* textLabel1;
    ImageCanvas* ThrusterCurrentMeterCanvas1;
    ImageCanvas* ThrusterCurrentMeterCanvas0;
    QGroupBox* groupBox1;
    QGroupBox* groupBox9_3;
    QLineEdit* field_int_press;
    ImageCanvas* IPressureCanvas;
    QCheckBox* ExtPressAuto;
    QSpinBox* IntPressMax;
    QSpinBox* IntPressMin;
    QSpinBox* ExtPressMax;
    QSpinBox* ExtPressMin;
    QGroupBox* groupBox6;
    ImageCanvas* CompassCanvas;
    QLineEdit* field_heading;
    QCheckBox* IntPressAuto;
    QGroupBox* groupBox10;
    ImageCanvas* EPressureCanvas;
    QLineEdit* field_ext_press;
    ImageCanvas* ImageDisplay0_2_2;
    QLineEdit* desired_heading_field_2_3;
    QLineEdit* desired_heading_field_2_2;
    QGroupBox* groupBox10_2;
    QGroupBox* groupBox2;
    QButtonGroup* buttonGroup1;
    QRadioButton* radio_manual;
    QRadioButton* radio_auto;
    QLineEdit* desired_speed_field;
    QLineEdit* desired_depth_field;
    QLabel* textLabel2_3_2_2;
    QLabel* textLabel2_2_2_3_2_2;
    QLabel* textLabel2_2_2_3_2_2_2;
    QLineEdit* desired_heading_field;
    QGroupBox* groupBox4;
    QLabel* textLabel2_3_2_2_2;
    QLabel* textLabel2_3_2_2_2_2;
    QLineEdit* heading_output_field;
    QLineEdit* depth_output_field;
    QTabWidget* tabWidget3;
    QWidget* tab;
    QFrame* frame4;
    ImageCanvas* ImageDisplay1;
    ImageCanvas* ImageDisplay2;
    ImageCanvas* ImageDisplay3;
    ImageCanvas* ImageDisplay0;
    QWidget* TabPage;
    QFrame* frame4_2;
    ImageCanvas* MovementDisplay0;
    ImageCanvas* MovementDisplay1;
    QWidget* tab_2;
    QGroupBox* groupBox9_2;
    QLabel* textLabel2_3_2_2_3_3_2;
    QLabel* textLabel2_3_2_2_3_2_2;
    QLineEdit* field_depth_i;
    QLineEdit* field_depth_d;
    QLineEdit* field_depth_k;
    QLabel* textLabel2_3_2_2_3_2_4_2_2;
    QLabel* textLabel2_3_2_2_3_4_2;
    QLineEdit* field_depth_p;
    QGroupBox* groupBox11;
    QLabel* textLabel1_4_2;
    QLabel* textLabel1_4_3;
    QLabel* textLabel1_4_3_2;
    QLabel* textLabel1_4;
    QPushButton* recordButton;
    QPushButton* stopButton;
    QPushButton* eraseButton;
    QPushButton* saveButton;
    QLabel* textLabel1_4_4;
    QLCDNumber* dataLoggerLCD;
    QGroupBox* groupBox9;
    QLabel* textLabel2_3_2_2_3_2_4_2;
    QLabel* textLabel2_3_2_2_3_2;
    QLabel* textLabel2_3_2_2_3_3;
    QLabel* textLabel2_3_2_2_3_4;
    QLineEdit* field_heading_i;
    QLineEdit* field_heading_k;
    QLineEdit* field_heading_d;
    QLineEdit* field_heading_p;
    QWidget* TabPage_2;
    ImageCanvas* MapCanvas;
    QGroupBox* groupBox15;
    QLabel* textLabel1_5;
    QLineEdit* MapName;
    QPushButton* SaveMapBtn;
    QPushButton* LoadMapBtn;
    QGroupBox* groupBox12;
    QPushButton* PlatformBtn;
    QLabel* PlatformPic;
    QLabel* GatePic;
    QPushButton* GateBtn;
    QLabel* PipePic;
    QPushButton* PipeBtn;
    QLabel* BuoyPic;
    QPushButton* FlareBtn;
    QLabel* BinPic;
    QLabel* BarbwirePic;
    QLabel* MachineGunNestPic;
    QLabel* OctagonSurfacePic;
    QPushButton* BinBtn;
    QPushButton* OctagonSurfaceBtn;
    QPushButton* MachineGunNestBtn;
    QPushButton* BarbwireBtn;
    QGroupBox* groupBox13;
    QComboBox* ObjectList;
    QLabel* textLabel2_3;
    QLineEdit* MapObjectY;
    QLabel* textLabel1_6;
    QLabel* textLabel2_4;
    QLineEdit* MapObjectX;
    QLineEdit* MapObjectVarX;
    QLineEdit* MapObjectVarY;
    QLabel* textLabel1_7;
    QPushButton* DeleteBtn;
    QPushButton* PlaceBtn;
    QWidget* TabPage_3;
    QPushButton* BriefcaseFoundBtn;
    QPushButton* BombingRunBtn;
    QPushButton* ContourFoundBoxesBtn;
    QPushButton* BarbwireDoneBtn;
    QPushButton* ContourFoundBarbwireBtn;
    QPushButton* FlareDoneBtn;
    QPushButton* GateFoundBtn;
    QPushButton* InitDoneBtn;
    QPushButton* GateDoneBtn;
    QPushButton* ContourFoundFlareBtn;
    QWidget* TabPage_4;
    QLabel* textLabel1_8;
    QLineEdit* colorFileLoadText;
    QPushButton* ColorPickLoadBut;
    QLabel* textLabel2_3_2_2_5;
    QLabel* textLabel2_3_2_2_4;
    QLabel* textLabel2_3_2_2_4_3;
    QLabel* textLabel2_3_2_2_7;
    QLabel* textLabel2_3_2_2_6;
    QLineEdit* v_mean_val;
    QLineEdit* h1__mean_val;
    QLineEdit* h2_mean_val;
    QLineEdit* s_mean_val;
    ImageCanvas* AvgColorImg;
    QLabel* textLabel2_3_2_2_4_3_2;
    QLineEdit* h1_std_val;
    QLineEdit* h2_std_val;
    QLineEdit* v_std_val;
    QLineEdit* s_std_val;
    ImageCanvas* ColorPickerImg;

    virtual void setSensorVotes( std::vector<ImageIceMod::SensorVote> votes );
    virtual void setCompositeHeading( int heading );
    virtual void setCompositeDepth( int depth );
    virtual void pushRectAngle( float a );
    virtual Image<PixRGB<byte> > makeCompassImage( int heading );
    virtual void setThrusterMeters( int zero, int one, int two, int three, int four, int five, int six, int seven );

public slots:
    virtual void pushFwdRectangle( ImageIceMod::QuadrilateralIce quad );
    virtual void pushRectRatio( float r );
    virtual void pushContourPoint( Point2D<int> p );
    virtual void setSalientPoint( Point2D<float> p );
    virtual void Image1Click( int a, int b, int c );
    virtual void init( ModelManager * mgr );
    virtual void setImage( Image<PixRGB<byte> > & img, std::string cameraID );
    virtual void ToggleCamera0();
    virtual void ToggleCamera1();
    virtual void ToggleCamera2();
    virtual void ToggleCamera3();
    virtual void ToggleCamera( std::string cameraID, bool active );
    virtual void registerCommunicator( nub::soft_ref<SeaBee3GUICommunicator> c );
    virtual void setJSAxis( int axis, float val );
    virtual void setSensorValues( int heading, int pitch, int roll, int intPressure, int extPressure, int headingValue, int depthValue );
    virtual void updateDesiredHeading();
    virtual void updateDesiredDepth();
    virtual void updateDesiredSpeed();
    virtual void updateHeadingPID();
    virtual void updateDepthPID();
    virtual void manualClicked();
    virtual void autoClicked();
    virtual void sendInitDone();
    virtual void sendGateFound();
    virtual void sendGateDone();
    virtual void sendContourFoundFlare();
    virtual void sendFlareDone();
    virtual void sendContourFoundBarbwire();
    virtual void sendBarbwireDone();
    virtual void sendContourFoundBoxes();
    virtual void sendBombingRunDone();
    virtual void sendBriefcaseFound();
    virtual void addPlatform();
    virtual void addGate();
    virtual void addPipe();
    virtual void addFlare();
    virtual void addBin();
    virtual void addBarbwire();
    virtual void addMachineGunNest();
    virtual void addOctagonSurface();
    virtual void selectObject( int index );
    virtual void refreshMapImage();
    virtual void moveObject();
    virtual void deleteObject();
    virtual void saveMap();
    virtual void loadMap();
    virtual void loadColorPickerImg();
    virtual void clickColorPickerImg( int x, int y, int but );

protected:
    std::vector<ImageIceMod::QuadrilateralIce> itsFwdRectangles;
    std::vector<Point2D<int> > itsRectangles;
    std::vector<MapperI::MapObject> mapObjects;
    SimpleMeter *currentMeter7;
    SimpleMeter *currentMeter6;
    SimpleMeter *currentMeter5;
    SimpleMeter *currentMeter4;
    SimpleMeter *currentMeter3;
    SimpleMeter *currentMeter2;
    SimpleMeter *currentMeter1;
    SimpleMeter *currentMeter0;
    nub::soft_ref<SeaBee3GUICommunicator> GUIComm;
    ModelManager *itsMgr;
    std::list<int> headingHist;
    std::list<int> extPressHist;
    std::list<int> intPressHist;
    IceUtil::Mutex itsDataMutex;
    int selectedIndex;
    Image<PixRGB<byte > > MapImage;
    Point2D<float> itsSalientPoint;
    std::list<float> itsRectAngles;
    std::list<float> itsRectRatios;
    int itsCompositeHeading;
    int itsCompositeDepth;
    std::vector<ImageIceMod::SensorVote> itsSensorVotes;
    int itsCurrentHeading;
    int itsCurrentDepth;
    Image<PixH2SV2<float> > itsConvert;


protected slots:
    virtual void languageChange();

private:
    QPixmap image0;
    QPixmap image1;
    QPixmap image2;
    QPixmap image3;
    QPixmap image4;
    QPixmap image5;
    QPixmap image6;
    QPixmap image7;
    QPixmap image8;

};

#endif // SEABEE3MAINDISPLAYFORM_H
