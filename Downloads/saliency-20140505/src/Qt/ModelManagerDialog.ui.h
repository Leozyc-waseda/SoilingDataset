/*!@file Qt/ModelManagerDialog.ui.h functions relating to model configuration
        dialog */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt/ModelManagerDialog.ui.h $
// $Id: ModelManagerDialog.ui.h 7095 2006-09-01 18:07:02Z rjpeters $

/****************************************************************************
** ui.h extension file, included from the uic-generated form implementation.
**
** If you wish to add, delete or rename functions or slots use
** Qt Designer which will update this file, preserving your code. Create an
** init() function in place of a constructor, and a destroy() function in
** place of a destructor.
*****************************************************************************/

void ModelManagerDialog::init( ModelManager & manager )
{
        mgr = &manager;
        pmap.clear();
        backupMap.clear();
        mgr->writeParamsTo( pmap );
        mgr->writeParamsTo( backupMap );
        listview->clear();
        listview->setDefaultRenameAction( QListView::Accept );
        QListViewItem *modelItem = new QListViewItem( listview, "model" );
        modelItem->setOpen( true );
        populate( pmap.getSubpmap( "model" ), modelItem );
        listview->triggerUpdate();
}

void ModelManagerDialog::populate( rutz::shared_ptr< ParamMap > pmp, QListViewItem *parent )
{
        ParamMap::key_iterator kitr = pmp->keys_begin();
        while( kitr != pmp->keys_end() )
        {
                QListViewItem *newItem = new QListViewItem( parent, *kitr, pmp->getStringParam( *kitr ) );
                newItem->setOpen( true );

                // if leaf, allow user to edit
                if( pmp->isLeaf( *kitr ) )
                        newItem->setRenameEnabled( 1, true );
                // otherwise, populate the submap
                else
                        populate( pmp->getSubpmap( *kitr ), newItem );

                ++kitr;
        }
}


void ModelManagerDialog::handleItemEdit( QListViewItem * item )
{
        int depth = item->depth();
        int d = depth;
        std::vector< QString > maps( d + 1 );
        maps[d] = item->text( 0 );
        QListViewItem *parent = item->parent();

        // find path to list view item
        while( d > 0 )
        {
                d = parent->depth();
                maps[d] = parent->text( 0 );
                parent = parent->parent();
        }
        rutz::shared_ptr< ParamMap > pm = pmap.getSubpmap( "model" );
        for( d = 1; d < depth; d++ )
        {
                pm = pm->getSubpmap( maps[d].ascii() );
        }

        // edit the item
        pm->replaceStringParam( item->text( 0 ).ascii(), item->text( 1 ).ascii() );

        // update the model manager
        mgr->readParamsFrom( pmap );
        pmap.clear();
        mgr->writeParamsTo( pmap );

        // refresh the list view
        listview->clear();
        QListViewItem *modelItem = new QListViewItem( listview, "model" );
        modelItem->setOpen( true );
        populate( pmap.getSubpmap( "model" ), modelItem );

        // put the edited item back in focus
        QListViewItem *edited = listview->firstChild();
        for( d = 0; d <= depth; d++ )
        {
                while( ( edited != 0 ) && ( edited->text( 0 ) != maps[d] ) )
                        edited = edited->nextSibling();
                if( d < depth )
                        edited = edited->firstChild();
        }
        if( edited )
        {
                listview->ensureItemVisible( edited );
                listview->clearSelection();
                edited->setSelected( true );
        }
        listview->triggerUpdate();
}


void ModelManagerDialog::handleWizardButton( void )
{
        mmw.init( *mgr );
        mmw.show();
        close();
}


void ModelManagerDialog::handleApplyButton( void )
{
        mgr->readParamsFrom( pmap );
        close();
}



void ModelManagerDialog::handleCancelButton( void )
{
        mgr->readParamsFrom( backupMap );
        close();
}
