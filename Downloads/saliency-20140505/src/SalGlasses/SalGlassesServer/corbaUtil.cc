/*!@file corbaUtil.cc */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the //
// University of Southern California (USC) and the iLab at USC.         //
// See http://iLab.usc.edu for information about this project.          //
// //////////////////////////////////////////////////////////////////// //
// Major portions of the iLab Neuromorphic Vision Toolkit are protected //
// under the U.S. patent ``Computation of Intrinsic Perceptual Saliency //
// in Visual Environments, and Applications'' by Christof Koch and      //
// Laurent Itti, California Institute of Technology, 2001 (patent       //
// pending; application number 09/912,225 filed July 23, 2001; see      //
// http://pair.uspto.gov/cgi-bin/final/home.pl for current status).     //
// //////////////////////////////////////////////////////////////////// //
// This file is part of the iLab Neuromorphic Vision C++ Toolkit.       //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is free software; you can   //
// redistribute it and/or modify it under the terms of the GNU General  //
// Public License as published by the Free Software Foundation; either  //
// version 2 of the License, or (at your option) any later version.     //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is distributed in the hope  //
// that it will be useful, but WITHOUT ANY WARRANTY; without even the   //
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR      //
// PURPOSE.  See the GNU General Public License for more details.       //
//                                                                      //
// You should have received a copy of the GNU General Public License    //
// along with the iLab Neuromorphic Vision C++ Toolkit; if not, write   //
// to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,   //
// Boston, MA 02111-1307 USA.                                           //
// //////////////////////////////////////////////////////////////////// //
//
// Primary maintainer for this file: Lior Elazary <lelazary@yahoo.com>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SalGlasses/SalGlassesServer/corbaUtil.cc $
// $Id: corbaUtil.cc 9108 2007-12-30 06:14:30Z rjpeters $
//

#include <stdio.h>
#include "corbaUtil.h"

//! Build an object reference for many object within a context
bool getMultiObjectRef(CORBA::ORB_ptr orb, const char *contextName, CORBA::Object_ptr *dst, int &nObj){
        printf("Looking for objects\n");
        CosNaming::NamingContext_var rootContext;

        try {
                // Obtain a reference to the root context of the Name service:
                CORBA::Object_var obj;
                obj = orb->resolve_initial_references("NameService");
                // Narrow the reference returned.
                rootContext = CosNaming::NamingContext::_narrow(obj);
                if( CORBA::is_nil(rootContext) ) {
                        printf("Failed to narrow the root naming context.\n");
                        return false;
                }
        }
        catch(CORBA::ORB::InvalidName& ex) {
                // This should not happen!
                printf("Service required is invalid [does not exist].\n");
                return false;
        }

        //find out how many objects we have to work with

        CosNaming::Name_var name = omniURI::stringToName(contextName);

        CORBA::Object_var obj = rootContext->resolve(name);

        CosNaming::NamingContext_var context;
        context = CosNaming::NamingContext::_narrow(obj);

        if (CORBA::is_nil(context)) {
                printf("No objects found\n");
                return false;
        }

        //get all the objects in the context

        CosNaming::BindingIterator_var bi;
        CosNaming::BindingList_var bl;
        CosNaming::Binding_var b;

        context->list(0, bl, bi);

        if (CORBA::is_nil(bi)){
                printf("No objects found\n");
                return false;
        }

        nObj=0;
        while (bi->next_one(b)){

                CosNaming::Name name_comp = b->binding_name;

                char* name = omniURI::nameToString(name_comp);
                printf("Binding to %s ... ", name);
                delete name;

                CORBA::Object_ptr ref = context->resolve(name_comp);
                dst[nObj] = ref;

                if (!CORBA::is_nil(dst[nObj])){
                        nObj++;
                        printf("Done\n");
                } else {
                        printf("Fail\n");
                }

        }
        bi->destroy();

        return true;
}

//////////////////////////////////////////////////////////////////////

CORBA::Boolean
bindObjectToName(CORBA::ORB_ptr orb, CORBA::Object_ptr objref,
                 const char* contextId, const char* contextKind,
                 const char* objectPrefix, CosNaming::Name& objectName)
{
        CosNaming::NamingContext_var rootContext;

        try {
                // Obtain a reference to the root context of the Name service:
                CORBA::Object_var obj;
                obj = orb->resolve_initial_references("NameService");

                if( CORBA::is_nil(obj) ) {
                        printf("Obj is null.\n");
                        return 0;
                }

                // Narrow the reference returned.
                rootContext = CosNaming::NamingContext::_narrow(obj);
                if( CORBA::is_nil(rootContext) ) {
                        printf("Failed to narrow the root naming context.\n");
                        return 0;
                }
        }
        catch(CORBA::ORB::InvalidName& ex) {
                // This should not happen!
                printf("Service required is invalid [does not exist].\n");
                return 0;
        }

        try {
                // Bind a context called "test" to the root context:

                CosNaming::Name contextName;
                contextName.length(1);
                contextName[0].id   = (const char*)contextId;   // string copied
                contextName[0].kind = (const char*)contextKind; // string copied
                // Note on kind: The kind field is used to indicate the type
                // of the object. This is to avoid conventions such as that used
                // by files (name.type -- e.g. test.ps = postscript etc.)
                CosNaming::NamingContext_var testContext;
                try {
                        // Bind the context to root.
                        testContext = rootContext->bind_new_context(contextName);
                }
                catch(CosNaming::NamingContext::AlreadyBound& ex) {
                        // If the context already exists, this exception will be raised.
                        // In this case, just resolve the name and assign testContext
                        // to the object returned:
                        CORBA::Object_var obj;
                        obj = rootContext->resolve(contextName);
                        testContext = CosNaming::NamingContext::_narrow(obj);
                        if( CORBA::is_nil(testContext) ) {
                                printf("Failed to narrow naming context.\n");
                                return 0;
                        }
                }

                // Bind objref with name Echo to the testContext:
                objectName.length(1);


                bool bound = false;
                char CmapID[100];
                for (int i=0; i<100 && !bound; i++) {
                        sprintf(CmapID, "%s_%i", objectPrefix, i);
                        printf("Binding object %s\n", CmapID);
                        objectName[0].id   = (const char*) CmapID;   // string copied
                        objectName[0].kind = (const char*) "Object"; // string copied

                        bound = true;
                        try {
                                testContext->bind(objectName, objref);
                        }
                        catch(CosNaming::NamingContext::AlreadyBound& ex) {
                                //testContext->rebind(objectName, objref);
                                bound = false;
                        }

                }

                if (!bound){
                        printf("Can not bind object\n");
                        return 0;
                } else {
                }


                // Amendment: When using OrbixNames, it is necessary to first try bind
                // and then rebind, as rebind on it's own will throw a NotFoundexception if
                // the Name has not already been bound. [This is incorrect behaviour -
                // it should just bind].
        }
        catch(CORBA::COMM_FAILURE& ex) {
                printf("Caught system exception COMM_FAILURE -- unable to contact the naming service.\n");
                return 0;
        }
        catch(CORBA::SystemException&) {
                printf("Caught a CORBA::SystemException while using the naming service.\n");
                return 0;
        }

        return 1;
}

// unbind the object from the name server
void unbindObject (CORBA::ORB_ptr orb, const char* contextId,
    const char* contextKind,
    CosNaming::Name &objectName){
  CosNaming::NamingContext_var rootContext;

        try {
                // Obtain a reference to the root context of the Name service:
                CORBA::Object_var obj;
                obj = orb->resolve_initial_references("NameService");

                if( CORBA::is_nil(obj) ) {
                        printf("Obj is null.\n");
                        return;
                }

                // Narrow the reference returned.
                rootContext = CosNaming::NamingContext::_narrow(obj);
                if( CORBA::is_nil(rootContext) ) {
                        printf("Failed to narrow the root naming context.\n");
                        return;
                }
        }
        catch(CORBA::ORB::InvalidName& ex) {
                // This should not happen!
                printf("Service required is invalid [does not exist].\n");
                return;
        }

        CosNaming::Name contextName;
        contextName.length(2);
        contextName[0].id   = (const char*)contextId;
        contextName[0].kind = (const char*)contextKind;
        contextName[1] = objectName[0];

        rootContext->unbind(contextName);
}
