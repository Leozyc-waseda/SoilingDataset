/****************************************************************************
** Form implementation generated from reading ui file 'Qt/SeaBee3GUI2.ui'
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "Qt/ui/SeaBee3GUI2.h"

#include <qvariant.h>
#include <qlineedit.h>
#include <qgroupbox.h>
#include <qlabel.h>
#include <qcheckbox.h>
#include <qpushbutton.h>
#include <qtabwidget.h>
#include <qwidget.h>
#include <qframe.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include <qaction.h>
#include <qmenubar.h>
#include <qpopupmenu.h>
#include <qtoolbar.h>
#include "Qt/ImageCanvas.h"
#include "Qt/SeaBee3GUI2.ui.h"

/*
 *  Constructs a SeaBee3MainDisplayForm as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 */
SeaBee3MainDisplayForm::SeaBee3MainDisplayForm( QWidget* parent, const char* name, WFlags fl )
    : QMainWindow( parent, name, fl )
{
    (void)statusBar();
    if ( !name )
        setName( "SeaBee3MainDisplayForm" );
    setPaletteForegroundColor( QColor( 255, 255, 255 ) );
    setPaletteBackgroundColor( QColor( 0, 0, 0 ) );
    setCentralWidget( new QWidget( this, "qt_central_widget" ) );

    desired_heading_field_2_3 = new QLineEdit( centralWidget(), "desired_heading_field_2_3" );
    desired_heading_field_2_3->setEnabled( TRUE );
    desired_heading_field_2_3->setGeometry( QRect( 430, 190, 60, 21 ) );

    desired_heading_field_2_2 = new QLineEdit( centralWidget(), "desired_heading_field_2_2" );
    desired_heading_field_2_2->setEnabled( TRUE );
    desired_heading_field_2_2->setGeometry( QRect( 430, 260, 60, 21 ) );

    groupBox2_3_3_3 = new QGroupBox( centralWidget(), "groupBox2_3_3_3" );
    groupBox2_3_3_3->setEnabled( TRUE );
    groupBox2_3_3_3->setGeometry( QRect( 760, 550, 220, 170 ) );
    groupBox2_3_3_3->setPaletteForegroundColor( QColor( 255, 255, 255 ) );
    QPalette pal;
    QColorGroup cg;
    cg.setColor( QColorGroup::Foreground, white );
    cg.setColor( QColorGroup::Button, QColor( 221, 223, 228) );
    cg.setColor( QColorGroup::Light, white );
    cg.setColor( QColorGroup::Midlight, QColor( 238, 239, 241) );
    cg.setColor( QColorGroup::Dark, QColor( 110, 111, 114) );
    cg.setColor( QColorGroup::Mid, QColor( 147, 149, 152) );
    cg.setColor( QColorGroup::Text, white );
    cg.setColor( QColorGroup::BrightText, white );
    cg.setColor( QColorGroup::ButtonText, white );
    cg.setColor( QColorGroup::Base, black );
    cg.setColor( QColorGroup::Background, QColor( 44, 44, 44) );
    cg.setColor( QColorGroup::Shadow, black );
    cg.setColor( QColorGroup::Highlight, QColor( 0, 0, 128) );
    cg.setColor( QColorGroup::HighlightedText, white );
    cg.setColor( QColorGroup::Link, black );
    cg.setColor( QColorGroup::LinkVisited, black );
    pal.setActive( cg );
    cg.setColor( QColorGroup::Foreground, white );
    cg.setColor( QColorGroup::Button, QColor( 221, 223, 228) );
    cg.setColor( QColorGroup::Light, white );
    cg.setColor( QColorGroup::Midlight, QColor( 254, 254, 255) );
    cg.setColor( QColorGroup::Dark, QColor( 110, 111, 114) );
    cg.setColor( QColorGroup::Mid, QColor( 147, 149, 152) );
    cg.setColor( QColorGroup::Text, white );
    cg.setColor( QColorGroup::BrightText, white );
    cg.setColor( QColorGroup::ButtonText, white );
    cg.setColor( QColorGroup::Base, black );
    cg.setColor( QColorGroup::Background, QColor( 44, 44, 44) );
    cg.setColor( QColorGroup::Shadow, black );
    cg.setColor( QColorGroup::Highlight, QColor( 0, 0, 128) );
    cg.setColor( QColorGroup::HighlightedText, white );
    cg.setColor( QColorGroup::Link, QColor( 0, 0, 238) );
    cg.setColor( QColorGroup::LinkVisited, QColor( 82, 24, 139) );
    pal.setInactive( cg );
    cg.setColor( QColorGroup::Foreground, QColor( 128, 128, 128) );
    cg.setColor( QColorGroup::Button, QColor( 221, 223, 228) );
    cg.setColor( QColorGroup::Light, white );
    cg.setColor( QColorGroup::Midlight, QColor( 254, 254, 255) );
    cg.setColor( QColorGroup::Dark, QColor( 110, 111, 114) );
    cg.setColor( QColorGroup::Mid, QColor( 147, 149, 152) );
    cg.setColor( QColorGroup::Text, QColor( 128, 128, 128) );
    cg.setColor( QColorGroup::BrightText, white );
    cg.setColor( QColorGroup::ButtonText, QColor( 128, 128, 128) );
    cg.setColor( QColorGroup::Base, black );
    cg.setColor( QColorGroup::Background, QColor( 44, 44, 44) );
    cg.setColor( QColorGroup::Shadow, black );
    cg.setColor( QColorGroup::Highlight, QColor( 0, 0, 128) );
    cg.setColor( QColorGroup::HighlightedText, white );
    cg.setColor( QColorGroup::Link, QColor( 0, 0, 238) );
    cg.setColor( QColorGroup::LinkVisited, QColor( 82, 24, 139) );
    pal.setDisabled( cg );
    groupBox2_3_3_3->setPalette( pal );

    desired_speed_field_3_3_3 = new QLineEdit( groupBox2_3_3_3, "desired_speed_field_3_3_3" );
    desired_speed_field_3_3_3->setEnabled( TRUE );
    desired_speed_field_3_3_3->setGeometry( QRect( 320, 60, 60, 20 ) );

    desired_depth_field_3_3_3 = new QLineEdit( groupBox2_3_3_3, "desired_depth_field_3_3_3" );
    desired_depth_field_3_3_3->setEnabled( TRUE );
    desired_depth_field_3_3_3->setGeometry( QRect( 320, 40, 60, 20 ) );
    desired_depth_field_3_3_3->setPaletteForegroundColor( QColor( 0, 0, 0 ) );

    desired_heading_field_3_3_3 = new QLineEdit( groupBox2_3_3_3, "desired_heading_field_3_3_3" );
    desired_heading_field_3_3_3->setEnabled( TRUE );
    desired_heading_field_3_3_3->setGeometry( QRect( 320, 20, 60, 21 ) );

    textLabel2_3_2_2_3_5_2_3_4_2_2_2 = new QLabel( groupBox2_3_3_3, "textLabel2_3_2_2_3_5_2_3_4_2_2_2" );
    textLabel2_3_2_2_3_5_2_3_4_2_2_2->setEnabled( TRUE );
    textLabel2_3_2_2_3_5_2_3_4_2_2_2->setGeometry( QRect( 150, 70, 40, 22 ) );
    QFont textLabel2_3_2_2_3_5_2_3_4_2_2_2_font(  textLabel2_3_2_2_3_5_2_3_4_2_2_2->font() );
    textLabel2_3_2_2_3_5_2_3_4_2_2_2->setFont( textLabel2_3_2_2_3_5_2_3_4_2_2_2_font );
    textLabel2_3_2_2_3_5_2_3_4_2_2_2->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    textLabel2_3_2_2_3_5_2_3_4_2_2_2_2_2 = new QLabel( groupBox2_3_3_3, "textLabel2_3_2_2_3_5_2_3_4_2_2_2_2_2" );
    textLabel2_3_2_2_3_5_2_3_4_2_2_2_2_2->setEnabled( TRUE );
    textLabel2_3_2_2_3_5_2_3_4_2_2_2_2_2->setGeometry( QRect( 150, 111, 40, 22 ) );
    QFont textLabel2_3_2_2_3_5_2_3_4_2_2_2_2_2_font(  textLabel2_3_2_2_3_5_2_3_4_2_2_2_2_2->font() );
    textLabel2_3_2_2_3_5_2_3_4_2_2_2_2_2->setFont( textLabel2_3_2_2_3_5_2_3_4_2_2_2_2_2_font );
    textLabel2_3_2_2_3_5_2_3_4_2_2_2_2_2->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    textLabel2_3_2_2_3_5_2_3_4_2_2 = new QLabel( groupBox2_3_3_3, "textLabel2_3_2_2_3_5_2_3_4_2_2" );
    textLabel2_3_2_2_3_5_2_3_4_2_2->setEnabled( TRUE );
    textLabel2_3_2_2_3_5_2_3_4_2_2->setGeometry( QRect( 150, 50, 40, 22 ) );
    QFont textLabel2_3_2_2_3_5_2_3_4_2_2_font(  textLabel2_3_2_2_3_5_2_3_4_2_2->font() );
    textLabel2_3_2_2_3_5_2_3_4_2_2->setFont( textLabel2_3_2_2_3_5_2_3_4_2_2_font );
    textLabel2_3_2_2_3_5_2_3_4_2_2->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    textLabel2_3_2_2_3_5_2_3_4_2_2_2_2 = new QLabel( groupBox2_3_3_3, "textLabel2_3_2_2_3_5_2_3_4_2_2_2_2" );
    textLabel2_3_2_2_3_5_2_3_4_2_2_2_2->setEnabled( TRUE );
    textLabel2_3_2_2_3_5_2_3_4_2_2_2_2->setGeometry( QRect( 150, 91, 40, 22 ) );
    QFont textLabel2_3_2_2_3_5_2_3_4_2_2_2_2_font(  textLabel2_3_2_2_3_5_2_3_4_2_2_2_2->font() );
    textLabel2_3_2_2_3_5_2_3_4_2_2_2_2->setFont( textLabel2_3_2_2_3_5_2_3_4_2_2_2_2_font );
    textLabel2_3_2_2_3_5_2_3_4_2_2_2_2->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_2 = new QLabel( groupBox2_3_3_3, "textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_2" );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_2->setEnabled( TRUE );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_2->setGeometry( QRect( 10, 140, 71, 22 ) );
    QFont textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_2_font(  textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_2->font() );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_2->setFont( textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_2_font );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_2->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    textLabel2_3_2_2_3_5_2_3_4_2 = new QLabel( groupBox2_3_3_3, "textLabel2_3_2_2_3_5_2_3_4_2" );
    textLabel2_3_2_2_3_5_2_3_4_2->setEnabled( TRUE );
    textLabel2_3_2_2_3_5_2_3_4_2->setGeometry( QRect( 150, 29, 40, 22 ) );
    QFont textLabel2_3_2_2_3_5_2_3_4_2_font(  textLabel2_3_2_2_3_5_2_3_4_2->font() );
    textLabel2_3_2_2_3_5_2_3_4_2->setFont( textLabel2_3_2_2_3_5_2_3_4_2_font );
    textLabel2_3_2_2_3_5_2_3_4_2->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    textLabel2_2_2_3_2_2_3_2_2_3_3_2 = new QLabel( groupBox2_3_3_3, "textLabel2_2_2_3_2_2_3_2_2_3_3_2" );
    textLabel2_2_2_3_2_2_3_2_2_3_3_2->setEnabled( TRUE );
    textLabel2_2_2_3_2_2_3_2_2_3_3_2->setGeometry( QRect( 0, 50, 80, 22 ) );
    QFont textLabel2_2_2_3_2_2_3_2_2_3_3_2_font(  textLabel2_2_2_3_2_2_3_2_2_3_3_2->font() );
    textLabel2_2_2_3_2_2_3_2_2_3_3_2->setFont( textLabel2_2_2_3_2_2_3_2_2_3_3_2_font );
    textLabel2_2_2_3_2_2_3_2_2_3_3_2->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    textLabel2_2_2_3_2_2_2_2_2_2_3_3 = new QLabel( groupBox2_3_3_3, "textLabel2_2_2_3_2_2_2_2_2_2_3_3" );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3->setEnabled( TRUE );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3->setGeometry( QRect( 20, 70, 60, 22 ) );
    QFont textLabel2_2_2_3_2_2_2_2_2_2_3_3_font(  textLabel2_2_2_3_2_2_2_2_2_2_3_3->font() );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3->setFont( textLabel2_2_2_3_2_2_2_2_2_2_3_3_font );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2 = new QLabel( groupBox2_3_3_3, "textLabel2_2_2_3_2_2_2_2_2_2_3_3_2" );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2->setEnabled( TRUE );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2->setGeometry( QRect( 10, 90, 71, 22 ) );
    QFont textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_font(  textLabel2_2_2_3_2_2_2_2_2_2_3_3_2->font() );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2->setFont( textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_font );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2 = new QLabel( groupBox2_3_3_3, "textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2" );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2->setEnabled( TRUE );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2->setGeometry( QRect( 10, 111, 71, 22 ) );
    QFont textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_font(  textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2->font() );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2->setFont( textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_font );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    textLabel2_3_2_2_3_5_2_3_4_2_2_2_2_2_2 = new QLabel( groupBox2_3_3_3, "textLabel2_3_2_2_3_5_2_3_4_2_2_2_2_2_2" );
    textLabel2_3_2_2_3_5_2_3_4_2_2_2_2_2_2->setEnabled( TRUE );
    textLabel2_3_2_2_3_5_2_3_4_2_2_2_2_2_2->setGeometry( QRect( 160, 140, 46, 22 ) );
    QFont textLabel2_3_2_2_3_5_2_3_4_2_2_2_2_2_2_font(  textLabel2_3_2_2_3_5_2_3_4_2_2_2_2_2_2->font() );
    textLabel2_3_2_2_3_5_2_3_4_2_2_2_2_2_2->setFont( textLabel2_3_2_2_3_5_2_3_4_2_2_2_2_2_2_font );
    textLabel2_3_2_2_3_5_2_3_4_2_2_2_2_2_2->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_2_2 = new QLabel( groupBox2_3_3_3, "textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_2_2" );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_2_2->setEnabled( TRUE );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_2_2->setGeometry( QRect( 110, 10, 20, 16 ) );
    QFont textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_2_2_font(  textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_2_2->font() );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_2_2->setFont( textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_2_2_font );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_2_2->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    itsFwdRetinaMsgField = new QLineEdit( groupBox2_3_3_3, "itsFwdRetinaMsgField" );
    itsFwdRetinaMsgField->setEnabled( TRUE );
    itsFwdRetinaMsgField->setGeometry( QRect( 90, 29, 60, 21 ) );
    itsFwdRetinaMsgField->setPaletteForegroundColor( QColor( 32, 253, 0 ) );
    QFont itsFwdRetinaMsgField_font(  itsFwdRetinaMsgField->font() );
    itsFwdRetinaMsgField_font.setFamily( "Sawasdee" );
    itsFwdRetinaMsgField_font.setPointSize( 10 );
    itsFwdRetinaMsgField_font.setBold( TRUE );
    itsFwdRetinaMsgField->setFont( itsFwdRetinaMsgField_font );
    itsFwdRetinaMsgField->setAlignment( int( QLineEdit::AlignHCenter ) );

    itsBeeStemMsgField = new QLineEdit( groupBox2_3_3_3, "itsBeeStemMsgField" );
    itsBeeStemMsgField->setEnabled( TRUE );
    itsBeeStemMsgField->setGeometry( QRect( 90, 70, 60, 21 ) );
    itsBeeStemMsgField->setPaletteForegroundColor( QColor( 32, 253, 0 ) );
    QFont itsBeeStemMsgField_font(  itsBeeStemMsgField->font() );
    itsBeeStemMsgField_font.setFamily( "Sawasdee" );
    itsBeeStemMsgField_font.setPointSize( 10 );
    itsBeeStemMsgField_font.setBold( TRUE );
    itsBeeStemMsgField->setFont( itsBeeStemMsgField_font );
    itsBeeStemMsgField->setAlignment( int( QLineEdit::AlignHCenter ) );

    desired_heading_field_2_4_2_3_3_2_2_2 = new QLineEdit( groupBox2_3_3_3, "desired_heading_field_2_4_2_3_3_2_2_2" );
    desired_heading_field_2_4_2_3_3_2_2_2->setEnabled( TRUE );
    desired_heading_field_2_4_2_3_3_2_2_2->setGeometry( QRect( 90, 90, 60, 21 ) );
    desired_heading_field_2_4_2_3_3_2_2_2->setPaletteForegroundColor( QColor( 32, 253, 0 ) );
    QFont desired_heading_field_2_4_2_3_3_2_2_2_font(  desired_heading_field_2_4_2_3_3_2_2_2->font() );
    desired_heading_field_2_4_2_3_3_2_2_2_font.setFamily( "Sawasdee" );
    desired_heading_field_2_4_2_3_3_2_2_2_font.setPointSize( 10 );
    desired_heading_field_2_4_2_3_3_2_2_2_font.setBold( TRUE );
    desired_heading_field_2_4_2_3_3_2_2_2->setFont( desired_heading_field_2_4_2_3_3_2_2_2_font );
    desired_heading_field_2_4_2_3_3_2_2_2->setAlignment( int( QLineEdit::AlignHCenter ) );

    desired_heading_field_2_4_2_3_3_2_2_2_2_2 = new QLineEdit( groupBox2_3_3_3, "desired_heading_field_2_4_2_3_3_2_2_2_2_2" );
    desired_heading_field_2_4_2_3_3_2_2_2_2_2->setEnabled( TRUE );
    desired_heading_field_2_4_2_3_3_2_2_2_2_2->setGeometry( QRect( 90, 140, 60, 21 ) );
    desired_heading_field_2_4_2_3_3_2_2_2_2_2->setPaletteForegroundColor( QColor( 32, 253, 0 ) );
    QFont desired_heading_field_2_4_2_3_3_2_2_2_2_2_font(  desired_heading_field_2_4_2_3_3_2_2_2_2_2->font() );
    desired_heading_field_2_4_2_3_3_2_2_2_2_2_font.setFamily( "Sawasdee" );
    desired_heading_field_2_4_2_3_3_2_2_2_2_2_font.setPointSize( 10 );
    desired_heading_field_2_4_2_3_3_2_2_2_2_2_font.setBold( TRUE );
    desired_heading_field_2_4_2_3_3_2_2_2_2_2->setFont( desired_heading_field_2_4_2_3_3_2_2_2_2_2_font );
    desired_heading_field_2_4_2_3_3_2_2_2_2_2->setAlignment( int( QLineEdit::AlignHCenter ) );

    itsVisionMsgField = new QLineEdit( groupBox2_3_3_3, "itsVisionMsgField" );
    itsVisionMsgField->setEnabled( TRUE );
    itsVisionMsgField->setGeometry( QRect( 90, 110, 60, 21 ) );
    itsVisionMsgField->setPaletteForegroundColor( QColor( 32, 253, 0 ) );
    QFont itsVisionMsgField_font(  itsVisionMsgField->font() );
    itsVisionMsgField_font.setFamily( "Sawasdee" );
    itsVisionMsgField_font.setPointSize( 10 );
    itsVisionMsgField_font.setBold( TRUE );
    itsVisionMsgField->setFont( itsVisionMsgField_font );
    itsVisionMsgField->setAlignment( int( QLineEdit::AlignHCenter ) );

    itsDwnRetinaMsgField = new QLineEdit( groupBox2_3_3_3, "itsDwnRetinaMsgField" );
    itsDwnRetinaMsgField->setEnabled( TRUE );
    itsDwnRetinaMsgField->setGeometry( QRect( 90, 49, 60, 21 ) );
    itsDwnRetinaMsgField->setPaletteForegroundColor( QColor( 32, 253, 0 ) );
    QFont itsDwnRetinaMsgField_font(  itsDwnRetinaMsgField->font() );
    itsDwnRetinaMsgField_font.setFamily( "Sawasdee" );
    itsDwnRetinaMsgField_font.setPointSize( 10 );
    itsDwnRetinaMsgField_font.setBold( TRUE );
    itsDwnRetinaMsgField->setFont( itsDwnRetinaMsgField_font );
    itsDwnRetinaMsgField->setAlignment( int( QLineEdit::AlignHCenter ) );

    textLabel2_2_2_3_2_2_3_2_2_3_3 = new QLabel( groupBox2_3_3_3, "textLabel2_2_2_3_2_2_3_2_2_3_3" );
    textLabel2_2_2_3_2_2_3_2_2_3_3->setEnabled( TRUE );
    textLabel2_2_2_3_2_2_3_2_2_3_3->setGeometry( QRect( 0, 30, 80, 22 ) );
    QFont textLabel2_2_2_3_2_2_3_2_2_3_3_font(  textLabel2_2_2_3_2_2_3_2_2_3_3->font() );
    textLabel2_2_2_3_2_2_3_2_2_3_3->setFont( textLabel2_2_2_3_2_2_3_2_2_3_3_font );
    textLabel2_2_2_3_2_2_3_2_2_3_3->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    groupBox10_2 = new QGroupBox( centralWidget(), "groupBox10_2" );
    groupBox10_2->setEnabled( TRUE );
    groupBox10_2->setGeometry( QRect( 10, 550, 730, 200 ) );
    groupBox10_2->setPaletteBackgroundColor( QColor( 0, 0, 0 ) );

    groupBox2_3_3_2_2 = new QGroupBox( groupBox10_2, "groupBox2_3_3_2_2" );
    groupBox2_3_3_2_2->setEnabled( TRUE );
    groupBox2_3_3_2_2->setGeometry( QRect( 10, 0, 230, 170 ) );
    groupBox2_3_3_2_2->setPaletteForegroundColor( QColor( 255, 255, 255 ) );
    cg.setColor( QColorGroup::Foreground, white );
    cg.setColor( QColorGroup::Button, QColor( 221, 223, 228) );
    cg.setColor( QColorGroup::Light, white );
    cg.setColor( QColorGroup::Midlight, QColor( 238, 239, 241) );
    cg.setColor( QColorGroup::Dark, QColor( 110, 111, 114) );
    cg.setColor( QColorGroup::Mid, QColor( 147, 149, 152) );
    cg.setColor( QColorGroup::Text, white );
    cg.setColor( QColorGroup::BrightText, white );
    cg.setColor( QColorGroup::ButtonText, white );
    cg.setColor( QColorGroup::Base, black );
    cg.setColor( QColorGroup::Background, QColor( 44, 44, 44) );
    cg.setColor( QColorGroup::Shadow, black );
    cg.setColor( QColorGroup::Highlight, QColor( 0, 0, 128) );
    cg.setColor( QColorGroup::HighlightedText, white );
    cg.setColor( QColorGroup::Link, black );
    cg.setColor( QColorGroup::LinkVisited, black );
    pal.setActive( cg );
    cg.setColor( QColorGroup::Foreground, white );
    cg.setColor( QColorGroup::Button, QColor( 221, 223, 228) );
    cg.setColor( QColorGroup::Light, white );
    cg.setColor( QColorGroup::Midlight, QColor( 254, 254, 255) );
    cg.setColor( QColorGroup::Dark, QColor( 110, 111, 114) );
    cg.setColor( QColorGroup::Mid, QColor( 147, 149, 152) );
    cg.setColor( QColorGroup::Text, white );
    cg.setColor( QColorGroup::BrightText, white );
    cg.setColor( QColorGroup::ButtonText, white );
    cg.setColor( QColorGroup::Base, black );
    cg.setColor( QColorGroup::Background, QColor( 44, 44, 44) );
    cg.setColor( QColorGroup::Shadow, black );
    cg.setColor( QColorGroup::Highlight, QColor( 0, 0, 128) );
    cg.setColor( QColorGroup::HighlightedText, white );
    cg.setColor( QColorGroup::Link, QColor( 0, 0, 238) );
    cg.setColor( QColorGroup::LinkVisited, QColor( 82, 24, 139) );
    pal.setInactive( cg );
    cg.setColor( QColorGroup::Foreground, QColor( 128, 128, 128) );
    cg.setColor( QColorGroup::Button, QColor( 221, 223, 228) );
    cg.setColor( QColorGroup::Light, white );
    cg.setColor( QColorGroup::Midlight, QColor( 254, 254, 255) );
    cg.setColor( QColorGroup::Dark, QColor( 110, 111, 114) );
    cg.setColor( QColorGroup::Mid, QColor( 147, 149, 152) );
    cg.setColor( QColorGroup::Text, QColor( 128, 128, 128) );
    cg.setColor( QColorGroup::BrightText, white );
    cg.setColor( QColorGroup::ButtonText, QColor( 128, 128, 128) );
    cg.setColor( QColorGroup::Base, black );
    cg.setColor( QColorGroup::Background, QColor( 44, 44, 44) );
    cg.setColor( QColorGroup::Shadow, black );
    cg.setColor( QColorGroup::Highlight, QColor( 0, 0, 128) );
    cg.setColor( QColorGroup::HighlightedText, white );
    cg.setColor( QColorGroup::Link, QColor( 0, 0, 238) );
    cg.setColor( QColorGroup::LinkVisited, QColor( 82, 24, 139) );
    pal.setDisabled( cg );
    groupBox2_3_3_2_2->setPalette( pal );

    desired_speed_field_3_3_2_2 = new QLineEdit( groupBox2_3_3_2_2, "desired_speed_field_3_3_2_2" );
    desired_speed_field_3_3_2_2->setEnabled( TRUE );
    desired_speed_field_3_3_2_2->setGeometry( QRect( 320, 60, 60, 20 ) );

    desired_depth_field_3_3_2_2 = new QLineEdit( groupBox2_3_3_2_2, "desired_depth_field_3_3_2_2" );
    desired_depth_field_3_3_2_2->setEnabled( TRUE );
    desired_depth_field_3_3_2_2->setGeometry( QRect( 320, 40, 60, 20 ) );
    desired_depth_field_3_3_2_2->setPaletteForegroundColor( QColor( 0, 0, 0 ) );

    desired_heading_field_3_3_2_2 = new QLineEdit( groupBox2_3_3_2_2, "desired_heading_field_3_3_2_2" );
    desired_heading_field_3_3_2_2->setEnabled( TRUE );
    desired_heading_field_3_3_2_2->setGeometry( QRect( 320, 20, 60, 21 ) );

    textLabel2_3_2_2_3_5_2_3_3 = new QLabel( groupBox2_3_3_2_2, "textLabel2_3_2_2_3_5_2_3_3" );
    textLabel2_3_2_2_3_5_2_3_3->setEnabled( TRUE );
    textLabel2_3_2_2_3_5_2_3_3->setGeometry( QRect( 46, 22, 54, 20 ) );
    QFont textLabel2_3_2_2_3_5_2_3_3_font(  textLabel2_3_2_2_3_5_2_3_3->font() );
    textLabel2_3_2_2_3_5_2_3_3->setFont( textLabel2_3_2_2_3_5_2_3_3_font );
    textLabel2_3_2_2_3_5_2_3_3->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    textLabel2_3_2_2_3_5_2_3_2_2 = new QLabel( groupBox2_3_3_2_2, "textLabel2_3_2_2_3_5_2_3_2_2" );
    textLabel2_3_2_2_3_5_2_3_2_2->setEnabled( TRUE );
    textLabel2_3_2_2_3_5_2_3_2_2->setGeometry( QRect( 10, 40, 20, 22 ) );
    QFont textLabel2_3_2_2_3_5_2_3_2_2_font(  textLabel2_3_2_2_3_5_2_3_2_2->font() );
    textLabel2_3_2_2_3_5_2_3_2_2->setFont( textLabel2_3_2_2_3_5_2_3_2_2_font );
    textLabel2_3_2_2_3_5_2_3_2_2->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    textLabel2_2_2_3_2_2_3_2_2_3_2_2 = new QLabel( groupBox2_3_3_2_2, "textLabel2_2_2_3_2_2_3_2_2_3_2_2" );
    textLabel2_2_2_3_2_2_3_2_2_3_2_2->setEnabled( TRUE );
    textLabel2_2_2_3_2_2_3_2_2_3_2_2->setGeometry( QRect( 10, 60, 16, 22 ) );
    QFont textLabel2_2_2_3_2_2_3_2_2_3_2_2_font(  textLabel2_2_2_3_2_2_3_2_2_3_2_2->font() );
    textLabel2_2_2_3_2_2_3_2_2_3_2_2->setFont( textLabel2_2_2_3_2_2_3_2_2_3_2_2_font );
    textLabel2_2_2_3_2_2_3_2_2_3_2_2->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    textLabel2_2_2_3_2_2_2_2_2_2_3_2_2 = new QLabel( groupBox2_3_3_2_2, "textLabel2_2_2_3_2_2_2_2_2_2_3_2_2" );
    textLabel2_2_2_3_2_2_2_2_2_2_3_2_2->setEnabled( TRUE );
    textLabel2_2_2_3_2_2_2_2_2_2_3_2_2->setGeometry( QRect( 10, 80, 20, 22 ) );
    QFont textLabel2_2_2_3_2_2_2_2_2_2_3_2_2_font(  textLabel2_2_2_3_2_2_2_2_2_2_3_2_2->font() );
    textLabel2_2_2_3_2_2_2_2_2_2_3_2_2->setFont( textLabel2_2_2_3_2_2_2_2_2_2_3_2_2_font );
    textLabel2_2_2_3_2_2_2_2_2_2_3_2_2->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    textLabel2_2_2_3_2_2_2_2_2_2_3_2_2_3 = new QLabel( groupBox2_3_3_2_2, "textLabel2_2_2_3_2_2_2_2_2_2_3_2_2_3" );
    textLabel2_2_2_3_2_2_2_2_2_2_3_2_2_3->setEnabled( TRUE );
    textLabel2_2_2_3_2_2_2_2_2_2_3_2_2_3->setGeometry( QRect( 120, 82, 20, 22 ) );
    QFont textLabel2_2_2_3_2_2_2_2_2_2_3_2_2_3_font(  textLabel2_2_2_3_2_2_2_2_2_2_3_2_2_3->font() );
    textLabel2_2_2_3_2_2_2_2_2_2_3_2_2_3->setFont( textLabel2_2_2_3_2_2_2_2_2_2_3_2_2_3_font );
    textLabel2_2_2_3_2_2_2_2_2_2_3_2_2_3->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    textLabel2_2_2_3_2_2_3_2_2_3_2_2_3 = new QLabel( groupBox2_3_3_2_2, "textLabel2_2_2_3_2_2_3_2_2_3_2_2_3" );
    textLabel2_2_2_3_2_2_3_2_2_3_2_2_3->setEnabled( TRUE );
    textLabel2_2_2_3_2_2_3_2_2_3_2_2_3->setGeometry( QRect( 120, 62, 16, 22 ) );
    QFont textLabel2_2_2_3_2_2_3_2_2_3_2_2_3_font(  textLabel2_2_2_3_2_2_3_2_2_3_2_2_3->font() );
    textLabel2_2_2_3_2_2_3_2_2_3_2_2_3->setFont( textLabel2_2_2_3_2_2_3_2_2_3_2_2_3_font );
    textLabel2_2_2_3_2_2_3_2_2_3_2_2_3->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    desired_heading_field_2_4_2_3_2_2_3 = new QLineEdit( groupBox2_3_3_2_2, "desired_heading_field_2_4_2_3_2_2_3" );
    desired_heading_field_2_4_2_3_2_2_3->setEnabled( TRUE );
    desired_heading_field_2_4_2_3_2_2_3->setGeometry( QRect( 150, 43, 60, 21 ) );
    desired_heading_field_2_4_2_3_2_2_3->setAlignment( int( QLineEdit::AlignHCenter ) );

    desired_depth_field_2_2_2_3_2_2_3 = new QLineEdit( groupBox2_3_3_2_2, "desired_depth_field_2_2_2_3_2_2_3" );
    desired_depth_field_2_2_2_3_2_2_3->setEnabled( TRUE );
    desired_depth_field_2_2_2_3_2_2_3->setGeometry( QRect( 150, 62, 60, 21 ) );
    desired_depth_field_2_2_2_3_2_2_3->setPaletteForegroundColor( QColor( 255, 255, 255 ) );
    desired_depth_field_2_2_2_3_2_2_3->setPaletteBackgroundColor( QColor( 0, 0, 0 ) );
    desired_depth_field_2_2_2_3_2_2_3->setAlignment( int( QLineEdit::AlignHCenter ) );

    desired_speed_field_2_2_2_3_2_2_3 = new QLineEdit( groupBox2_3_3_2_2, "desired_speed_field_2_2_2_3_2_2_3" );
    desired_speed_field_2_2_2_3_2_2_3->setEnabled( TRUE );
    desired_speed_field_2_2_2_3_2_2_3->setGeometry( QRect( 150, 81, 60, 20 ) );
    desired_speed_field_2_2_2_3_2_2_3->setAlignment( int( QLineEdit::AlignHCenter ) );

    textLabel2_3_2_2_3_5_2_3_3_3 = new QLabel( groupBox2_3_3_2_2, "textLabel2_3_2_2_3_5_2_3_3_3" );
    textLabel2_3_2_2_3_5_2_3_3_3->setEnabled( TRUE );
    textLabel2_3_2_2_3_5_2_3_3_3->setGeometry( QRect( 160, 20, 45, 20 ) );
    QFont textLabel2_3_2_2_3_5_2_3_3_3_font(  textLabel2_3_2_2_3_5_2_3_3_3->font() );
    textLabel2_3_2_2_3_5_2_3_3_3->setFont( textLabel2_3_2_2_3_5_2_3_3_3_font );
    textLabel2_3_2_2_3_5_2_3_3_3->setAlignment( int( QLabel::WordBreak | QLabel::AlignVCenter | QLabel::AlignRight ) );

    textLabel2_3_2_2_3_5_2_3_2_2_3 = new QLabel( groupBox2_3_3_2_2, "textLabel2_3_2_2_3_5_2_3_2_2_3" );
    textLabel2_3_2_2_3_5_2_3_2_2_3->setEnabled( TRUE );
    textLabel2_3_2_2_3_5_2_3_2_2_3->setGeometry( QRect( 120, 42, 20, 22 ) );
    QFont textLabel2_3_2_2_3_5_2_3_2_2_3_font(  textLabel2_3_2_2_3_5_2_3_2_2_3->font() );
    textLabel2_3_2_2_3_5_2_3_2_2_3->setFont( textLabel2_3_2_2_3_5_2_3_2_2_3_font );
    textLabel2_3_2_2_3_5_2_3_2_2_3->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    desired_heading_field_2_4_2_3_2_2 = new QLineEdit( groupBox2_3_3_2_2, "desired_heading_field_2_4_2_3_2_2" );
    desired_heading_field_2_4_2_3_2_2->setEnabled( TRUE );
    desired_heading_field_2_4_2_3_2_2->setGeometry( QRect( 40, 41, 60, 21 ) );
    desired_heading_field_2_4_2_3_2_2->setAlignment( int( QLineEdit::AlignHCenter ) );

    desired_depth_field_2_2_2_3_2_2 = new QLineEdit( groupBox2_3_3_2_2, "desired_depth_field_2_2_2_3_2_2" );
    desired_depth_field_2_2_2_3_2_2->setEnabled( TRUE );
    desired_depth_field_2_2_2_3_2_2->setGeometry( QRect( 40, 60, 60, 21 ) );
    desired_depth_field_2_2_2_3_2_2->setPaletteForegroundColor( QColor( 255, 255, 255 ) );
    desired_depth_field_2_2_2_3_2_2->setPaletteBackgroundColor( QColor( 0, 0, 0 ) );
    desired_depth_field_2_2_2_3_2_2->setAlignment( int( QLineEdit::AlignHCenter ) );

    desired_speed_field_2_2_2_3_2_2 = new QLineEdit( groupBox2_3_3_2_2, "desired_speed_field_2_2_2_3_2_2" );
    desired_speed_field_2_2_2_3_2_2->setEnabled( TRUE );
    desired_speed_field_2_2_2_3_2_2->setGeometry( QRect( 40, 79, 60, 20 ) );
    desired_speed_field_2_2_2_3_2_2->setAlignment( int( QLineEdit::AlignHCenter ) );

    EPressureCanvas_2 = new ImageCanvas( groupBox2_3_3_2_2, "EPressureCanvas_2" );
    EPressureCanvas_2->setEnabled( TRUE );
    EPressureCanvas_2->setGeometry( QRect( 10, 110, 100, 50 ) );

    itsDepthPIDImageDisplay = new ImageCanvas( groupBox2_3_3_2_2, "itsDepthPIDImageDisplay" );
    itsDepthPIDImageDisplay->setEnabled( TRUE );
    itsDepthPIDImageDisplay->setGeometry( QRect( 120, 110, 100, 50 ) );

    groupBox2_3_3 = new QGroupBox( groupBox10_2, "groupBox2_3_3" );
    groupBox2_3_3->setEnabled( TRUE );
    groupBox2_3_3->setGeometry( QRect( 470, 0, 250, 170 ) );
    groupBox2_3_3->setPaletteForegroundColor( QColor( 255, 255, 255 ) );
    cg.setColor( QColorGroup::Foreground, white );
    cg.setColor( QColorGroup::Button, QColor( 221, 223, 228) );
    cg.setColor( QColorGroup::Light, white );
    cg.setColor( QColorGroup::Midlight, QColor( 238, 239, 241) );
    cg.setColor( QColorGroup::Dark, QColor( 110, 111, 114) );
    cg.setColor( QColorGroup::Mid, QColor( 147, 149, 152) );
    cg.setColor( QColorGroup::Text, white );
    cg.setColor( QColorGroup::BrightText, white );
    cg.setColor( QColorGroup::ButtonText, white );
    cg.setColor( QColorGroup::Base, black );
    cg.setColor( QColorGroup::Background, QColor( 44, 44, 44) );
    cg.setColor( QColorGroup::Shadow, black );
    cg.setColor( QColorGroup::Highlight, QColor( 0, 0, 128) );
    cg.setColor( QColorGroup::HighlightedText, white );
    cg.setColor( QColorGroup::Link, black );
    cg.setColor( QColorGroup::LinkVisited, black );
    pal.setActive( cg );
    cg.setColor( QColorGroup::Foreground, white );
    cg.setColor( QColorGroup::Button, QColor( 221, 223, 228) );
    cg.setColor( QColorGroup::Light, white );
    cg.setColor( QColorGroup::Midlight, QColor( 254, 254, 255) );
    cg.setColor( QColorGroup::Dark, QColor( 110, 111, 114) );
    cg.setColor( QColorGroup::Mid, QColor( 147, 149, 152) );
    cg.setColor( QColorGroup::Text, white );
    cg.setColor( QColorGroup::BrightText, white );
    cg.setColor( QColorGroup::ButtonText, white );
    cg.setColor( QColorGroup::Base, black );
    cg.setColor( QColorGroup::Background, QColor( 44, 44, 44) );
    cg.setColor( QColorGroup::Shadow, black );
    cg.setColor( QColorGroup::Highlight, QColor( 0, 0, 128) );
    cg.setColor( QColorGroup::HighlightedText, white );
    cg.setColor( QColorGroup::Link, QColor( 0, 0, 238) );
    cg.setColor( QColorGroup::LinkVisited, QColor( 82, 24, 139) );
    pal.setInactive( cg );
    cg.setColor( QColorGroup::Foreground, QColor( 128, 128, 128) );
    cg.setColor( QColorGroup::Button, QColor( 221, 223, 228) );
    cg.setColor( QColorGroup::Light, white );
    cg.setColor( QColorGroup::Midlight, QColor( 254, 254, 255) );
    cg.setColor( QColorGroup::Dark, QColor( 110, 111, 114) );
    cg.setColor( QColorGroup::Mid, QColor( 147, 149, 152) );
    cg.setColor( QColorGroup::Text, QColor( 128, 128, 128) );
    cg.setColor( QColorGroup::BrightText, white );
    cg.setColor( QColorGroup::ButtonText, QColor( 128, 128, 128) );
    cg.setColor( QColorGroup::Base, black );
    cg.setColor( QColorGroup::Background, QColor( 44, 44, 44) );
    cg.setColor( QColorGroup::Shadow, black );
    cg.setColor( QColorGroup::Highlight, QColor( 0, 0, 128) );
    cg.setColor( QColorGroup::HighlightedText, white );
    cg.setColor( QColorGroup::Link, QColor( 0, 0, 238) );
    cg.setColor( QColorGroup::LinkVisited, QColor( 82, 24, 139) );
    pal.setDisabled( cg );
    groupBox2_3_3->setPalette( pal );

    desired_speed_field_3_3 = new QLineEdit( groupBox2_3_3, "desired_speed_field_3_3" );
    desired_speed_field_3_3->setEnabled( TRUE );
    desired_speed_field_3_3->setGeometry( QRect( 320, 60, 60, 20 ) );

    desired_depth_field_3_3 = new QLineEdit( groupBox2_3_3, "desired_depth_field_3_3" );
    desired_depth_field_3_3->setEnabled( TRUE );
    desired_depth_field_3_3->setGeometry( QRect( 320, 40, 60, 20 ) );
    desired_depth_field_3_3->setPaletteForegroundColor( QColor( 0, 0, 0 ) );

    desired_heading_field_3_3 = new QLineEdit( groupBox2_3_3, "desired_heading_field_3_3" );
    desired_heading_field_3_3->setEnabled( TRUE );
    desired_heading_field_3_3->setGeometry( QRect( 320, 20, 60, 21 ) );

    checkBox3_4_2_2_3 = new QCheckBox( groupBox2_3_3, "checkBox3_4_2_2_3" );
    checkBox3_4_2_2_3->setGeometry( QRect( 10, 21, 70, 16 ) );

    pushButton1 = new QPushButton( groupBox2_3_3, "pushButton1" );
    pushButton1->setGeometry( QRect( 70, 140, 101, 21 ) );
    pushButton1->setPaletteBackgroundColor( QColor( 0, 0, 0 ) );

    textLabel2_2_2_3_2_2_2_2_2_2_3_4 = new QLabel( groupBox2_3_3, "textLabel2_2_2_3_2_2_2_2_2_2_3_4" );
    textLabel2_2_2_3_2_2_2_2_2_2_3_4->setEnabled( TRUE );
    textLabel2_2_2_3_2_2_2_2_2_2_3_4->setGeometry( QRect( 198, 50, 42, 20 ) );
    QFont textLabel2_2_2_3_2_2_2_2_2_2_3_4_font(  textLabel2_2_2_3_2_2_2_2_2_2_3_4->font() );
    textLabel2_2_2_3_2_2_2_2_2_2_3_4->setFont( textLabel2_2_2_3_2_2_2_2_2_2_3_4_font );
    textLabel2_2_2_3_2_2_2_2_2_2_3_4->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    textLabel2_2_2_3_2_2_2_2_2_2_3 = new QLabel( groupBox2_3_3, "textLabel2_2_2_3_2_2_2_2_2_2_3" );
    textLabel2_2_2_3_2_2_2_2_2_2_3->setEnabled( TRUE );
    textLabel2_2_2_3_2_2_2_2_2_2_3->setGeometry( QRect( 130, 50, 50, 16 ) );
    QFont textLabel2_2_2_3_2_2_2_2_2_2_3_font(  textLabel2_2_2_3_2_2_2_2_2_2_3->font() );
    textLabel2_2_2_3_2_2_2_2_2_2_3->setFont( textLabel2_2_2_3_2_2_2_2_2_2_3_font );
    textLabel2_2_2_3_2_2_2_2_2_2_3->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    itsStrafeAxisImageDisplay_2 = new ImageCanvas( groupBox2_3_3, "itsStrafeAxisImageDisplay_2" );
    itsStrafeAxisImageDisplay_2->setEnabled( TRUE );
    itsStrafeAxisImageDisplay_2->setGeometry( QRect( 190, 70, 50, 50 ) );

    itsDepthAxisImageDisplay = new ImageCanvas( groupBox2_3_3, "itsDepthAxisImageDisplay" );
    itsDepthAxisImageDisplay->setEnabled( TRUE );
    itsDepthAxisImageDisplay->setGeometry( QRect( 70, 70, 50, 50 ) );

    textLabel2_2_2_3_2_2_3_2_2_3 = new QLabel( groupBox2_3_3, "textLabel2_2_2_3_2_2_3_2_2_3" );
    textLabel2_2_2_3_2_2_3_2_2_3->setEnabled( TRUE );
    textLabel2_2_2_3_2_2_3_2_2_3->setGeometry( QRect( 80, 50, 41, 20 ) );
    QFont textLabel2_2_2_3_2_2_3_2_2_3_font(  textLabel2_2_2_3_2_2_3_2_2_3->font() );
    textLabel2_2_2_3_2_2_3_2_2_3->setFont( textLabel2_2_2_3_2_2_3_2_2_3_font );
    textLabel2_2_2_3_2_2_3_2_2_3->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    itsHeadingAxisImageDisplay = new ImageCanvas( groupBox2_3_3, "itsHeadingAxisImageDisplay" );
    itsHeadingAxisImageDisplay->setEnabled( TRUE );
    itsHeadingAxisImageDisplay->setGeometry( QRect( 10, 70, 50, 50 ) );

    textLabel2_3_2_2_3_5_2_3 = new QLabel( groupBox2_3_3, "textLabel2_3_2_2_3_5_2_3" );
    textLabel2_3_2_2_3_5_2_3->setEnabled( TRUE );
    textLabel2_3_2_2_3_5_2_3->setGeometry( QRect( 10, 50, 54, 20 ) );
    QFont textLabel2_3_2_2_3_5_2_3_font(  textLabel2_3_2_2_3_5_2_3->font() );
    textLabel2_3_2_2_3_5_2_3->setFont( textLabel2_3_2_2_3_5_2_3_font );
    textLabel2_3_2_2_3_5_2_3->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    itsStrafeAxisImageDisplay = new ImageCanvas( groupBox2_3_3, "itsStrafeAxisImageDisplay" );
    itsStrafeAxisImageDisplay->setEnabled( TRUE );
    itsStrafeAxisImageDisplay->setGeometry( QRect( 130, 70, 50, 50 ) );

    desired_speed_field_2_2_2_3_2_3_2 = new QLineEdit( groupBox2_3_3, "desired_speed_field_2_2_2_3_2_3_2" );
    desired_speed_field_2_2_2_3_2_3_2->setEnabled( FALSE );
    desired_speed_field_2_2_2_3_2_3_2->setGeometry( QRect( 140, 20, 100, 21 ) );
    desired_speed_field_2_2_2_3_2_3_2->setPaletteForegroundColor( QColor( 253, 0, 0 ) );
    desired_speed_field_2_2_2_3_2_3_2->setPaletteBackgroundColor( QColor( 0, 0, 0 ) );
    QFont desired_speed_field_2_2_2_3_2_3_2_font(  desired_speed_field_2_2_2_3_2_3_2->font() );
    desired_speed_field_2_2_2_3_2_3_2_font.setPointSize( 8 );
    desired_speed_field_2_2_2_3_2_3_2->setFont( desired_speed_field_2_2_2_3_2_3_2_font );
    desired_speed_field_2_2_2_3_2_3_2->setAlignment( int( QLineEdit::AlignHCenter ) );

    groupBox2_3_3_2 = new QGroupBox( groupBox10_2, "groupBox2_3_3_2" );
    groupBox2_3_3_2->setEnabled( TRUE );
    groupBox2_3_3_2->setGeometry( QRect( 250, 0, 210, 170 ) );
    groupBox2_3_3_2->setPaletteForegroundColor( QColor( 255, 255, 255 ) );
    cg.setColor( QColorGroup::Foreground, white );
    cg.setColor( QColorGroup::Button, QColor( 221, 223, 228) );
    cg.setColor( QColorGroup::Light, white );
    cg.setColor( QColorGroup::Midlight, QColor( 238, 239, 241) );
    cg.setColor( QColorGroup::Dark, QColor( 110, 111, 114) );
    cg.setColor( QColorGroup::Mid, QColor( 147, 149, 152) );
    cg.setColor( QColorGroup::Text, white );
    cg.setColor( QColorGroup::BrightText, white );
    cg.setColor( QColorGroup::ButtonText, white );
    cg.setColor( QColorGroup::Base, black );
    cg.setColor( QColorGroup::Background, QColor( 44, 44, 44) );
    cg.setColor( QColorGroup::Shadow, black );
    cg.setColor( QColorGroup::Highlight, QColor( 0, 0, 128) );
    cg.setColor( QColorGroup::HighlightedText, white );
    cg.setColor( QColorGroup::Link, black );
    cg.setColor( QColorGroup::LinkVisited, black );
    pal.setActive( cg );
    cg.setColor( QColorGroup::Foreground, white );
    cg.setColor( QColorGroup::Button, QColor( 221, 223, 228) );
    cg.setColor( QColorGroup::Light, white );
    cg.setColor( QColorGroup::Midlight, QColor( 254, 254, 255) );
    cg.setColor( QColorGroup::Dark, QColor( 110, 111, 114) );
    cg.setColor( QColorGroup::Mid, QColor( 147, 149, 152) );
    cg.setColor( QColorGroup::Text, white );
    cg.setColor( QColorGroup::BrightText, white );
    cg.setColor( QColorGroup::ButtonText, white );
    cg.setColor( QColorGroup::Base, black );
    cg.setColor( QColorGroup::Background, QColor( 44, 44, 44) );
    cg.setColor( QColorGroup::Shadow, black );
    cg.setColor( QColorGroup::Highlight, QColor( 0, 0, 128) );
    cg.setColor( QColorGroup::HighlightedText, white );
    cg.setColor( QColorGroup::Link, QColor( 0, 0, 238) );
    cg.setColor( QColorGroup::LinkVisited, QColor( 82, 24, 139) );
    pal.setInactive( cg );
    cg.setColor( QColorGroup::Foreground, QColor( 128, 128, 128) );
    cg.setColor( QColorGroup::Button, QColor( 221, 223, 228) );
    cg.setColor( QColorGroup::Light, white );
    cg.setColor( QColorGroup::Midlight, QColor( 254, 254, 255) );
    cg.setColor( QColorGroup::Dark, QColor( 110, 111, 114) );
    cg.setColor( QColorGroup::Mid, QColor( 147, 149, 152) );
    cg.setColor( QColorGroup::Text, QColor( 128, 128, 128) );
    cg.setColor( QColorGroup::BrightText, white );
    cg.setColor( QColorGroup::ButtonText, QColor( 128, 128, 128) );
    cg.setColor( QColorGroup::Base, black );
    cg.setColor( QColorGroup::Background, QColor( 44, 44, 44) );
    cg.setColor( QColorGroup::Shadow, black );
    cg.setColor( QColorGroup::Highlight, QColor( 0, 0, 128) );
    cg.setColor( QColorGroup::HighlightedText, white );
    cg.setColor( QColorGroup::Link, QColor( 0, 0, 238) );
    cg.setColor( QColorGroup::LinkVisited, QColor( 82, 24, 139) );
    pal.setDisabled( cg );
    groupBox2_3_3_2->setPalette( pal );

    desired_speed_field_3_3_2 = new QLineEdit( groupBox2_3_3_2, "desired_speed_field_3_3_2" );
    desired_speed_field_3_3_2->setEnabled( TRUE );
    desired_speed_field_3_3_2->setGeometry( QRect( 320, 60, 60, 20 ) );

    desired_depth_field_3_3_2 = new QLineEdit( groupBox2_3_3_2, "desired_depth_field_3_3_2" );
    desired_depth_field_3_3_2->setEnabled( TRUE );
    desired_depth_field_3_3_2->setGeometry( QRect( 320, 40, 60, 20 ) );
    desired_depth_field_3_3_2->setPaletteForegroundColor( QColor( 0, 0, 0 ) );

    desired_heading_field_3_3_2 = new QLineEdit( groupBox2_3_3_2, "desired_heading_field_3_3_2" );
    desired_heading_field_3_3_2->setEnabled( TRUE );
    desired_heading_field_3_3_2->setGeometry( QRect( 320, 20, 60, 21 ) );

    checkBox3_4_2_2_3_2 = new QCheckBox( groupBox2_3_3_2, "checkBox3_4_2_2_3_2" );
    checkBox3_4_2_2_3_2->setGeometry( QRect( 10, 20, 121, 21 ) );

    textLabel2_3_2_2_3_5_2_3_2 = new QLabel( groupBox2_3_3_2, "textLabel2_3_2_2_3_5_2_3_2" );
    textLabel2_3_2_2_3_5_2_3_2->setEnabled( TRUE );
    textLabel2_3_2_2_3_5_2_3_2->setGeometry( QRect( 10, 40, 105, 22 ) );
    QFont textLabel2_3_2_2_3_5_2_3_2_font(  textLabel2_3_2_2_3_5_2_3_2->font() );
    textLabel2_3_2_2_3_5_2_3_2->setFont( textLabel2_3_2_2_3_5_2_3_2_font );
    textLabel2_3_2_2_3_5_2_3_2->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    desired_speed_field_2_2_2_3_2 = new QLineEdit( groupBox2_3_3_2, "desired_speed_field_2_2_2_3_2" );
    desired_speed_field_2_2_2_3_2->setEnabled( TRUE );
    desired_speed_field_2_2_2_3_2->setGeometry( QRect( 120, 80, 60, 20 ) );
    desired_speed_field_2_2_2_3_2->setAlignment( int( QLineEdit::AlignHCenter ) );

    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_3 = new QLabel( groupBox2_3_3_2, "textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_3" );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_3->setEnabled( TRUE );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_3->setGeometry( QRect( 11, 120, 105, 22 ) );
    QFont textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_3_font(  textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_3->font() );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_3->setFont( textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_3_font );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_3->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    desired_depth_field_2_2_2_3_2 = new QLineEdit( groupBox2_3_3_2, "desired_depth_field_2_2_2_3_2" );
    desired_depth_field_2_2_2_3_2->setEnabled( TRUE );
    desired_depth_field_2_2_2_3_2->setGeometry( QRect( 120, 61, 60, 21 ) );
    desired_depth_field_2_2_2_3_2->setPaletteForegroundColor( QColor( 255, 255, 255 ) );
    desired_depth_field_2_2_2_3_2->setPaletteBackgroundColor( QColor( 0, 0, 0 ) );
    desired_depth_field_2_2_2_3_2->setAlignment( int( QLineEdit::AlignHCenter ) );

    desired_heading_field_2_4_2_3_2 = new QLineEdit( groupBox2_3_3_2, "desired_heading_field_2_4_2_3_2" );
    desired_heading_field_2_4_2_3_2->setEnabled( TRUE );
    desired_heading_field_2_4_2_3_2->setGeometry( QRect( 120, 41, 60, 21 ) );
    desired_heading_field_2_4_2_3_2->setAlignment( int( QLineEdit::AlignHCenter ) );

    textLabel2_2_2_3_2_2_3_2_2_3_2 = new QLabel( groupBox2_3_3_2, "textLabel2_2_2_3_2_2_3_2_2_3_2" );
    textLabel2_2_2_3_2_2_3_2_2_3_2->setEnabled( TRUE );
    textLabel2_2_2_3_2_2_3_2_2_3_2->setGeometry( QRect( 20, 60, 94, 22 ) );
    QFont textLabel2_2_2_3_2_2_3_2_2_3_2_font(  textLabel2_2_2_3_2_2_3_2_2_3_2->font() );
    textLabel2_2_2_3_2_2_3_2_2_3_2->setFont( textLabel2_2_2_3_2_2_3_2_2_3_2_font );
    textLabel2_2_2_3_2_2_3_2_2_3_2->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    textLabel2_2_2_3_2_2_2_2_2_2_3_2 = new QLabel( groupBox2_3_3_2, "textLabel2_2_2_3_2_2_2_2_2_2_3_2" );
    textLabel2_2_2_3_2_2_2_2_2_2_3_2->setEnabled( TRUE );
    textLabel2_2_2_3_2_2_2_2_2_2_3_2->setGeometry( QRect( 20, 80, 93, 22 ) );
    QFont textLabel2_2_2_3_2_2_2_2_2_2_3_2_font(  textLabel2_2_2_3_2_2_2_2_2_2_3_2->font() );
    textLabel2_2_2_3_2_2_2_2_2_2_3_2->setFont( textLabel2_2_2_3_2_2_2_2_2_2_3_2_font );
    textLabel2_2_2_3_2_2_2_2_2_2_3_2->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_3_2 = new QLabel( groupBox2_3_3_2, "textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_3_2" );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_3_2->setEnabled( TRUE );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_3_2->setGeometry( QRect( 0, 140, 110, 22 ) );
    QFont textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_3_2_font(  textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_3_2->font() );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_3_2->setFont( textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_3_2_font );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_3_2->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    itsHeadingOutputField = new QLineEdit( groupBox2_3_3_2, "itsHeadingOutputField" );
    itsHeadingOutputField->setEnabled( TRUE );
    itsHeadingOutputField->setGeometry( QRect( 120, 120, 60, 21 ) );
    itsHeadingOutputField->setPaletteForegroundColor( QColor( 32, 253, 0 ) );
    itsHeadingOutputField->setPaletteBackgroundColor( QColor( 0, 0, 0 ) );
    QFont itsHeadingOutputField_font(  itsHeadingOutputField->font() );
    itsHeadingOutputField_font.setFamily( "Sawasdee" );
    itsHeadingOutputField_font.setPointSize( 10 );
    itsHeadingOutputField_font.setBold( TRUE );
    itsHeadingOutputField->setFont( itsHeadingOutputField_font );
    itsHeadingOutputField->setAlignment( int( QLineEdit::AlignHCenter ) );

    itsDepthOutputField = new QLineEdit( groupBox2_3_3_2, "itsDepthOutputField" );
    itsDepthOutputField->setEnabled( TRUE );
    itsDepthOutputField->setGeometry( QRect( 120, 140, 60, 21 ) );
    itsDepthOutputField->setPaletteForegroundColor( QColor( 32, 253, 0 ) );
    itsDepthOutputField->setPaletteBackgroundColor( QColor( 0, 0, 0 ) );
    QFont itsDepthOutputField_font(  itsDepthOutputField->font() );
    itsDepthOutputField_font.setFamily( "Sawasdee" );
    itsDepthOutputField_font.setPointSize( 10 );
    itsDepthOutputField_font.setBold( TRUE );
    itsDepthOutputField->setFont( itsDepthOutputField_font );
    itsDepthOutputField->setAlignment( int( QLineEdit::AlignHCenter ) );

    groupBox1 = new QGroupBox( centralWidget(), "groupBox1" );
    groupBox1->setEnabled( TRUE );
    groupBox1->setGeometry( QRect( 740, 30, 270, 520 ) );
    groupBox1->setPaletteBackgroundColor( QColor( 0, 0, 0 ) );

    textLabel2_2_2_3_2_2_2_2_2_2_3_2_3 = new QLabel( groupBox1, "textLabel2_2_2_3_2_2_2_2_2_2_3_2_3" );
    textLabel2_2_2_3_2_2_2_2_2_2_3_2_3->setEnabled( TRUE );
    textLabel2_2_2_3_2_2_2_2_2_2_3_2_3->setGeometry( QRect( 130, 360, 70, 22 ) );
    QFont textLabel2_2_2_3_2_2_2_2_2_2_3_2_3_font(  textLabel2_2_2_3_2_2_2_2_2_2_3_2_3->font() );
    textLabel2_2_2_3_2_2_2_2_2_2_3_2_3->setFont( textLabel2_2_2_3_2_2_2_2_2_2_3_2_3_font );
    textLabel2_2_2_3_2_2_2_2_2_2_3_2_3->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    EPressureCanvas_3 = new ImageCanvas( groupBox1, "EPressureCanvas_3" );
    EPressureCanvas_3->setEnabled( TRUE );
    EPressureCanvas_3->setGeometry( QRect( 130, 390, 110, 120 ) );

    groupBox9_3 = new QGroupBox( groupBox1, "groupBox9_3" );
    groupBox9_3->setEnabled( TRUE );
    groupBox9_3->setGeometry( QRect( 0, 190, 250, 160 ) );
    groupBox9_3->setPaletteBackgroundColor( QColor( 44, 44, 44 ) );

    checkBox3_4 = new QCheckBox( groupBox9_3, "checkBox3_4" );
    checkBox3_4->setGeometry( QRect( 160, 19, 121, 21 ) );
    checkBox3_4->setPaletteForegroundColor( QColor( 32, 253, 0 ) );

    checkBox3_2_2 = new QCheckBox( groupBox9_3, "checkBox3_2_2" );
    checkBox3_2_2->setGeometry( QRect( 160, 40, 121, 21 ) );
    checkBox3_2_2->setPaletteForegroundColor( QColor( 33, 229, 255 ) );

    checkBox3_3_2 = new QCheckBox( groupBox9_3, "checkBox3_3_2" );
    checkBox3_3_2->setGeometry( QRect( 160, 60, 121, 21 ) );
    checkBox3_3_2->setPaletteForegroundColor( QColor( 255, 58, 58 ) );

    itsExternalPressureField = new QLineEdit( groupBox9_3, "itsExternalPressureField" );
    itsExternalPressureField->setGeometry( QRect( 160, 90, 70, 60 ) );
    itsExternalPressureField->setPaletteForegroundColor( QColor( 30, 253, 0 ) );
    itsExternalPressureField->setPaletteBackgroundColor( QColor( 0, 0, 0 ) );
    QFont itsExternalPressureField_font(  itsExternalPressureField->font() );
    itsExternalPressureField_font.setFamily( "Sawasdee" );
    itsExternalPressureField_font.setPointSize( 18 );
    itsExternalPressureField_font.setBold( TRUE );
    itsExternalPressureField->setFont( itsExternalPressureField_font );
    itsExternalPressureField->setFrame( FALSE );
    itsExternalPressureField->setCursorPosition( 3 );
    itsExternalPressureField->setAlignment( int( QLineEdit::AlignHCenter ) );

    ItsDepthImageDisplay = new ImageCanvas( groupBox9_3, "ItsDepthImageDisplay" );
    ItsDepthImageDisplay->setEnabled( TRUE );
    ItsDepthImageDisplay->setGeometry( QRect( 10, 20, 140, 130 ) );

    groupBox6 = new QGroupBox( groupBox1, "groupBox6" );
    groupBox6->setEnabled( TRUE );
    groupBox6->setGeometry( QRect( 0, 20, 250, 160 ) );
    groupBox6->setPaletteBackgroundColor( QColor( 44, 44, 44 ) );

    checkBox3 = new QCheckBox( groupBox6, "checkBox3" );
    checkBox3->setGeometry( QRect( 160, 19, 121, 21 ) );
    checkBox3->setPaletteForegroundColor( QColor( 32, 253, 0 ) );

    checkBox3_2 = new QCheckBox( groupBox6, "checkBox3_2" );
    checkBox3_2->setGeometry( QRect( 160, 40, 121, 21 ) );
    checkBox3_2->setPaletteForegroundColor( QColor( 33, 229, 255 ) );

    checkBox3_3 = new QCheckBox( groupBox6, "checkBox3_3" );
    checkBox3_3->setGeometry( QRect( 160, 60, 121, 21 ) );
    checkBox3_3->setPaletteForegroundColor( QColor( 255, 58, 58 ) );

    itsCompassImageDisplay = new ImageCanvas( groupBox6, "itsCompassImageDisplay" );
    itsCompassImageDisplay->setEnabled( TRUE );
    itsCompassImageDisplay->setGeometry( QRect( 10, 20, 140, 130 ) );

    itsCompassHeadingField = new QLineEdit( groupBox6, "itsCompassHeadingField" );
    itsCompassHeadingField->setGeometry( QRect( 160, 90, 70, 60 ) );
    itsCompassHeadingField->setPaletteForegroundColor( QColor( 30, 253, 0 ) );
    itsCompassHeadingField->setPaletteBackgroundColor( QColor( 0, 0, 0 ) );
    QFont itsCompassHeadingField_font(  itsCompassHeadingField->font() );
    itsCompassHeadingField_font.setFamily( "Sawasdee" );
    itsCompassHeadingField_font.setPointSize( 24 );
    itsCompassHeadingField_font.setBold( TRUE );
    itsCompassHeadingField->setFont( itsCompassHeadingField_font );
    itsCompassHeadingField->setFrame( FALSE );
    itsCompassHeadingField->setAlignment( int( QLineEdit::AlignHCenter ) );

    itsKillSwitchField = new QLineEdit( groupBox1, "itsKillSwitchField" );
    itsKillSwitchField->setEnabled( FALSE );
    itsKillSwitchField->setGeometry( QRect( 200, 360, 30, 21 ) );
    itsKillSwitchField->setPaletteForegroundColor( QColor( 253, 0, 0 ) );
    itsKillSwitchField->setPaletteBackgroundColor( QColor( 0, 0, 0 ) );
    itsKillSwitchField->setAlignment( int( QLineEdit::AlignHCenter ) );

    groupBox10 = new QGroupBox( groupBox1, "groupBox10" );
    groupBox10->setEnabled( TRUE );
    groupBox10->setGeometry( QRect( 0, 360, 120, 150 ) );

    itsInternalPressureField = new QLineEdit( groupBox10, "itsInternalPressureField" );
    itsInternalPressureField->setGeometry( QRect( 0, 120, 130, 35 ) );
    itsInternalPressureField->setPaletteForegroundColor( QColor( 30, 253, 0 ) );
    itsInternalPressureField->setPaletteBackgroundColor( QColor( 0, 0, 0 ) );
    QFont itsInternalPressureField_font(  itsInternalPressureField->font() );
    itsInternalPressureField_font.setFamily( "Sawasdee" );
    itsInternalPressureField_font.setPointSize( 20 );
    itsInternalPressureField_font.setBold( TRUE );
    itsInternalPressureField->setFont( itsInternalPressureField_font );
    itsInternalPressureField->setFrame( FALSE );
    itsInternalPressureField->setCursorPosition( 3 );
    itsInternalPressureField->setAlignment( int( QLineEdit::AlignHCenter ) );

    itsPressureImageDisplay = new ImageCanvas( groupBox10, "itsPressureImageDisplay" );
    itsPressureImageDisplay->setEnabled( TRUE );
    itsPressureImageDisplay->setGeometry( QRect( 10, 30, 100, 90 ) );

    tabWidget3 = new QTabWidget( centralWidget(), "tabWidget3" );
    tabWidget3->setEnabled( TRUE );
    tabWidget3->setGeometry( QRect( 10, 10, 730, 540 ) );
    tabWidget3->setPaletteForegroundColor( QColor( 104, 104, 104 ) );
    tabWidget3->setPaletteBackgroundColor( QColor( 0, 0, 0 ) );

    tab = new QWidget( tabWidget3, "tab" );

    groupBox15_2 = new QGroupBox( tab, "groupBox15_2" );
    groupBox15_2->setGeometry( QRect( 688, -28, 181, 171 ) );

    frame4 = new QFrame( tab, "frame4" );
    frame4->setEnabled( TRUE );
    frame4->setGeometry( QRect( 0, 10, 720, 495 ) );
    frame4->setPaletteBackgroundColor( QColor( 48, 48, 48 ) );
    frame4->setFrameShape( QFrame::StyledPanel );
    frame4->setFrameShadow( QFrame::Raised );

    itsDwnImgDisplay = new ImageCanvas( frame4, "itsDwnImgDisplay" );
    itsDwnImgDisplay->setEnabled( TRUE );
    itsDwnImgDisplay->setGeometry( QRect( 5, 250, 290, 240 ) );

    itsFwdImgDisplay = new ImageCanvas( frame4, "itsFwdImgDisplay" );
    itsFwdImgDisplay->setEnabled( TRUE );
    itsFwdImgDisplay->setGeometry( QRect( 5, 5, 290, 240 ) );

    itsPipeThreshCheck = new QCheckBox( frame4, "itsPipeThreshCheck" );
    itsPipeThreshCheck->setGeometry( QRect( 600, 259, 121, 21 ) );
    itsPipeThreshCheck->setPaletteForegroundColor( QColor( 255, 255, 255 ) );

    itsHoughVisionCheck = new QCheckBox( frame4, "itsHoughVisionCheck" );
    itsHoughVisionCheck->setGeometry( QRect( 600, 280, 121, 21 ) );
    itsHoughVisionCheck->setPaletteForegroundColor( QColor( 255, 255, 255 ) );

    itsFwdContourThreshCheck = new QCheckBox( frame4, "itsFwdContourThreshCheck" );
    itsFwdContourThreshCheck->setGeometry( QRect( 600, 50, 121, 21 ) );
    itsFwdContourThreshCheck->setPaletteForegroundColor( QColor( 255, 255, 255 ) );

    itsBuoyThreshCheck = new QCheckBox( frame4, "itsBuoyThreshCheck" );
    itsBuoyThreshCheck->setGeometry( QRect( 600, 8, 120, 21 ) );
    itsBuoyThreshCheck->setPaletteForegroundColor( QColor( 255, 255, 255 ) );

    itsFwdVisionDisplay = new ImageCanvas( frame4, "itsFwdVisionDisplay" );
    itsFwdVisionDisplay->setEnabled( TRUE );
    itsFwdVisionDisplay->setGeometry( QRect( 300, 5, 290, 240 ) );

    itsSaliencyVisionCheck = new QCheckBox( frame4, "itsSaliencyVisionCheck" );
    itsSaliencyVisionCheck->setGeometry( QRect( 600, 30, 121, 21 ) );
    itsSaliencyVisionCheck->setPaletteForegroundColor( QColor( 255, 255, 255 ) );

    itsDwnContourVisionCheck = new QCheckBox( frame4, "itsDwnContourVisionCheck" );
    itsDwnContourVisionCheck->setGeometry( QRect( 600, 300, 121, 21 ) );
    itsDwnContourVisionCheck->setPaletteForegroundColor( QColor( 255, 255, 255 ) );

    itsDwnVisionDisplay = new ImageCanvas( frame4, "itsDwnVisionDisplay" );
    itsDwnVisionDisplay->setEnabled( TRUE );
    itsDwnVisionDisplay->setGeometry( QRect( 300, 250, 290, 240 ) );
    tabWidget3->insertTab( tab, QString::fromLatin1("") );

    TabPage = new QWidget( tabWidget3, "TabPage" );
    tabWidget3->insertTab( TabPage, QString::fromLatin1("") );

    // toolbars

    languageChange();
    resize( QSize(1026, 740).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // signals and slots connections
    connect( itsBuoyThreshCheck, SIGNAL( toggled(bool) ), this, SLOT( updateBuoySegmentCheck(bool) ) );
    connect( itsSaliencyVisionCheck, SIGNAL( toggled(bool) ), this, SLOT( updateSalientPointCheck(bool) ) );
}

/*
 *  Destroys the object and frees any allocated resources
 */
SeaBee3MainDisplayForm::~SeaBee3MainDisplayForm()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void SeaBee3MainDisplayForm::languageChange()
{
    setCaption( tr( "SeaBee3 Main Display" ) );
    desired_heading_field_2_3->setText( tr( "0" ) );
    desired_heading_field_2_2->setText( tr( "0" ) );
    groupBox2_3_3_3->setTitle( tr( "ICE Messages" ) );
    desired_speed_field_3_3_3->setText( tr( "0" ) );
    desired_depth_field_3_3_3->setText( tr( "0" ) );
    desired_heading_field_3_3_3->setText( tr( "0" ) );
    textLabel2_3_2_2_3_5_2_3_4_2_2_2->setText( tr( "/ sec" ) );
    textLabel2_3_2_2_3_5_2_3_4_2_2_2_2_2->setText( tr( "/ sec" ) );
    textLabel2_3_2_2_3_5_2_3_4_2_2->setText( tr( "/ sec" ) );
    textLabel2_3_2_2_3_5_2_3_4_2_2_2_2->setText( tr( "/ sec" ) );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_2->setText( tr( "Sent:" ) );
    textLabel2_3_2_2_3_5_2_3_4_2->setText( tr( "/ sec" ) );
    textLabel2_2_2_3_2_2_3_2_2_3_3_2->setText( tr( "Dwn Retina:" ) );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3->setText( tr( "BeeStem:" ) );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2->setText( tr( "Movement:" ) );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2->setText( tr( "Vision Proc:" ) );
    textLabel2_3_2_2_3_5_2_3_4_2_2_2_2_2_2->setText( tr( "msg(s)" ) );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_2_2->setText( tr( "Rx" ) );
    itsFwdRetinaMsgField->setText( tr( "0" ) );
    itsBeeStemMsgField->setText( tr( "0" ) );
    desired_heading_field_2_4_2_3_3_2_2_2->setText( tr( "0" ) );
    desired_heading_field_2_4_2_3_3_2_2_2_2_2->setText( tr( "0" ) );
    itsVisionMsgField->setText( tr( "0" ) );
    itsDwnRetinaMsgField->setText( tr( "0" ) );
    textLabel2_2_2_3_2_2_3_2_2_3_3->setText( tr( "Fwd Retina:" ) );
    groupBox10_2->setTitle( tr( "Controls" ) );
    groupBox2_3_3_2_2->setTitle( tr( "PID Settings" ) );
    desired_speed_field_3_3_2_2->setText( tr( "0" ) );
    desired_depth_field_3_3_2_2->setText( tr( "0" ) );
    desired_heading_field_3_3_2_2->setText( tr( "0" ) );
    textLabel2_3_2_2_3_5_2_3_3->setText( tr( "Heading" ) );
    textLabel2_3_2_2_3_5_2_3_2_2->setText( tr( "P:" ) );
    textLabel2_2_2_3_2_2_3_2_2_3_2_2->setText( tr( "I:" ) );
    textLabel2_2_2_3_2_2_2_2_2_2_3_2_2->setText( tr( "D:" ) );
    textLabel2_2_2_3_2_2_2_2_2_2_3_2_2_3->setText( tr( "D:" ) );
    textLabel2_2_2_3_2_2_3_2_2_3_2_2_3->setText( tr( "I:" ) );
    desired_heading_field_2_4_2_3_2_2_3->setText( tr( "0" ) );
    desired_depth_field_2_2_2_3_2_2_3->setText( tr( "0" ) );
    desired_speed_field_2_2_2_3_2_2_3->setText( tr( "0" ) );
    textLabel2_3_2_2_3_5_2_3_3_3->setText( tr( "Depth" ) );
    textLabel2_3_2_2_3_5_2_3_2_2_3->setText( tr( "P:" ) );
    desired_heading_field_2_4_2_3_2_2->setText( tr( "0" ) );
    desired_depth_field_2_2_2_3_2_2->setText( tr( "0" ) );
    desired_speed_field_2_2_2_3_2_2->setText( tr( "0" ) );
    groupBox2_3_3->setTitle( tr( "Manual Control" ) );
    desired_speed_field_3_3->setText( tr( "0" ) );
    desired_depth_field_3_3->setText( tr( "0" ) );
    desired_heading_field_3_3->setText( tr( "0" ) );
    checkBox3_4_2_2_3->setText( tr( "Enable" ) );
    pushButton1->setText( tr( "Calibrate" ) );
    textLabel2_2_2_3_2_2_2_2_2_2_3_4->setText( tr( "Speed" ) );
    textLabel2_2_2_3_2_2_2_2_2_2_3->setText( tr( "Strafe" ) );
    textLabel2_2_2_3_2_2_3_2_2_3->setText( tr( "Depth" ) );
    textLabel2_3_2_2_3_5_2_3->setText( tr( "Heading" ) );
    desired_speed_field_2_2_2_3_2_3_2->setText( tr( "Disconnected" ) );
    groupBox2_3_3_2->setTitle( tr( "PID Control" ) );
    desired_speed_field_3_3_2->setText( tr( "0" ) );
    desired_depth_field_3_3_2->setText( tr( "0" ) );
    desired_heading_field_3_3_2->setText( tr( "0" ) );
    checkBox3_4_2_2_3_2->setText( tr( "Enable" ) );
    textLabel2_3_2_2_3_5_2_3_2->setText( tr( "Desired Heading:" ) );
    desired_speed_field_2_2_2_3_2->setText( tr( "0" ) );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_3->setText( tr( "Heading Output:" ) );
    desired_depth_field_2_2_2_3_2->setText( tr( "- -" ) );
    desired_heading_field_2_4_2_3_2->setText( tr( "- -" ) );
    textLabel2_2_2_3_2_2_3_2_2_3_2->setText( tr( "Desired Depth:" ) );
    textLabel2_2_2_3_2_2_2_2_2_2_3_2->setText( tr( "Desired Speed:" ) );
    textLabel2_2_2_3_2_2_2_2_2_2_3_3_2_2_3_2->setText( tr( "Depth Output:" ) );
    itsHeadingOutputField->setText( tr( "0" ) );
    itsDepthOutputField->setText( tr( "0" ) );
    groupBox1->setTitle( QString::null );
    textLabel2_2_2_3_2_2_2_2_2_2_3_2_3->setText( tr( "Kill Switch:" ) );
    groupBox9_3->setTitle( tr( "Depth" ) );
    checkBox3_4->setText( tr( "Desired" ) );
    checkBox3_2_2->setText( tr( "Saliency" ) );
    checkBox3_3_2->setText( tr( "Pipeline" ) );
    itsExternalPressureField->setText( tr( "- -" ) );
    groupBox6->setTitle( tr( "Heading" ) );
    checkBox3->setText( tr( "Desired" ) );
    checkBox3_2->setText( tr( "Saliency" ) );
    checkBox3_3->setText( tr( "Pipeline" ) );
    itsCompassHeadingField->setText( tr( "- -" ) );
    itsKillSwitchField->setText( tr( "ON" ) );
    groupBox10->setTitle( tr( "Internal Pressure" ) );
    itsInternalPressureField->setText( tr( "- -" ) );
    groupBox15_2->setTitle( tr( "groupBox15" ) );
    itsPipeThreshCheck->setText( tr( "Pipe Threshold" ) );
    itsHoughVisionCheck->setText( tr( "Hough" ) );
    itsFwdContourThreshCheck->setText( tr( "Contour" ) );
    itsBuoyThreshCheck->setText( tr( "Buoy Threshold" ) );
    itsSaliencyVisionCheck->setText( tr( "Saliency" ) );
    itsDwnContourVisionCheck->setText( tr( "Contour" ) );
    tabWidget3->changeTab( tab, tr( "Main" ) );
    tabWidget3->changeTab( TabPage, tr( "Localization" ) );
}

