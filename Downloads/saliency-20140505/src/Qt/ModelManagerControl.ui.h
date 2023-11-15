/*!@file Qt/ModelManagerControl.ui.h functions relating to model control
panel */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt/ModelManagerControl.ui.h $
// $Id: ModelManagerControl.ui.h 12962 2010-03-06 02:13:53Z irock $

/****************************************************************************
** ui.h extension file, included from the uic-generated form implementation.
**
** If you wish to add, delete or rename functions or slots use
** Qt Designer which will update this file, preserving your code. Create an
** init() function in place of a constructor, and a destroy() function in
** place of a destructor.
*****************************************************************************/


void ModelManagerControl::showConfigDialog( void )
{
        mmd.init( *mgr );
        mmd.show();
}


void ModelManagerControl::loadConfig( void )
{
        QString fname = QFileDialog::getOpenFileName( mgr->getModelParamString( "LoadConfigFile" ).c_str() );
        if( !fname.isNull() && !fname.isEmpty() )
                mgr->loadConfig( std::string( fname.ascii() ) );
}


void ModelManagerControl::saveConfig( void )
{
        QString fname = QFileDialog::getSaveFileName( mgr->getModelParamString( "SaveConfigFile" ).c_str() );
        if( !fname.isNull() && !fname.isEmpty() )
                mgr->saveConfig( std::string( fname.ascii() ) );
}


void ModelManagerControl::start_or_stop( void )
{
        if( mgr->started() )
        {
                startstopButton->setText( "Start" );
                configButton->setEnabled( true );
        *dorun = false;
        }
        else
        {
                startstopButton->setText( "Stop" );
                configButton->setEnabled( false );
        *dorun = true;
        }
}


void ModelManagerControl::init( ModelManager & manager, bool *dorun_ )
{
        mgr = &manager;
       dorun = dorun_;
        mmd.init( manager );
}


void ModelManagerControl::exitPressed( void )
{
        if( mgr->started() )
                mgr->stop();
        close();
}
