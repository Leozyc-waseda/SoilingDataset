/*!@file NeovisionII/nv2_label_reader.c */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
// by the University of Southern California (USC) and the iLab at USC.  //
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
// Primary maintainer for this file: Rob Peters <rjpeters at usc dot edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/nv2_label_reader.c $
// $Id: nv2_label_reader.c 9497 2008-03-19 04:19:53Z rjpeters $
//

#ifndef NEOVISIONII_NV2_LABEL_READER_C_DEFINED
#define NEOVISIONII_NV2_LABEL_READER_C_DEFINED

#include "NeovisionII/nv2_label_reader.h"

#include "NeovisionII/nv2_common.h"

// #include order matters here:
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct nv2_label_reader
{
        pthread_mutex_t m_label_lock;
        struct nv2_patch_label m_label;

        pthread_mutex_t m_patch_lock;
        struct nv2_image_patch m_patch;
        enum nv2_patch_send_result m_latest_patch_send_result;

        int m_label_reader_port;
        struct in_addr m_remote_patch_reader_in_addr;
        int m_remote_patch_reader_port;

        int m_reader_socket_fd;

        pthread_t m_label_reader_thread;
        pthread_t m_patch_sender_thread;
};

static void cleanup_fd(void* p)
{
        int* const fd = (int*) p;
        if (*fd >= 0)
        {
                close(*fd);
                *fd = -1;
        }
}

static void* run_label_reader(void* q)
{
        struct nv2_label_reader* p = (struct nv2_label_reader*) q;

        pthread_cleanup_push(&cleanup_fd, &p->m_reader_socket_fd);

        p->m_reader_socket_fd = socket(PF_INET, SOCK_STREAM, 0);
        if (p->m_reader_socket_fd == -1)
                nv2_fatal("socket() failed");

        {
                const int set_on = 1;
                if (setsockopt(p->m_reader_socket_fd,
                               SOL_SOCKET, SO_REUSEADDR,
                               &set_on, sizeof(set_on)) == -1)
                        nv2_warn("setsockopt() failed");
        }

        struct sockaddr_in name;
        name.sin_family = AF_INET;
        name.sin_addr.s_addr = htonl(INADDR_ANY); // accept from any host
        name.sin_port = htons(p->m_label_reader_port);
        if (bind(p->m_reader_socket_fd,
                 (struct sockaddr*)(&name), sizeof(name)) == -1)
                nv2_fatal("bind() failed");

        listen(p->m_reader_socket_fd, 5);

        int client_socket_fd = -1;
        pthread_cleanup_push(&cleanup_fd, &client_socket_fd);

        while (1)
        {
                struct sockaddr_in client_name;
                socklen_t client_name_len = sizeof(client_name);

                client_socket_fd =
                        accept(p->m_reader_socket_fd,
                               (struct sockaddr*)(&client_name),
                               &client_name_len);

                if (client_socket_fd < 0)
                {
                        // if accept() failed because
                        // p->m_reader_socket_fd was closed() and/or
                        // is less than 0, then that is because the
                        // main thread closed the socket to force us
                        // to shut down, so let's quit without any
                        // nv2_warn()
                        if (p->m_reader_socket_fd < 0)
                                break;

                        nv2_warn("accept() failed");
                        continue;
                }

                struct nv2_patch_label l;

                errno = 0;
                if (nv2_robust_read(client_socket_fd, &l, sizeof(l), NULL)
                    != sizeof(l))
                {
                        nv2_warn("read(nv2_patch_label) failed");
                        close(client_socket_fd);
                        client_socket_fd = -1;
                        continue;
                }

                l.protocol_version = ntohl(l.protocol_version);
                l.patch_id = ntohl(l.patch_id);
                l.confidence = ntohl(l.confidence);
                l.source[sizeof(l.source) - 1] = '\0';
                l.name[sizeof(l.name) - 1] = '\0';
                l.extra_info[sizeof(l.extra_info) - 1] = '\0';

                if (l.protocol_version != NV2_LABEL_PROTOCOL_VERSION)
                {
                        fprintf(stderr,
                                "wrong label protocol version "
                                "(got %u, expected %u)",
                                (unsigned int) l.protocol_version,
                                (unsigned int) NV2_LABEL_PROTOCOL_VERSION);
                        exit(-1);
                }

                // Enter this patch label into the next available
                // position in our queue:
                pthread_mutex_lock(&p->m_label_lock);
                p->m_label = l;
                pthread_mutex_unlock(&p->m_label_lock);

                close(client_socket_fd);
                client_socket_fd = -1;
        }

        pthread_cleanup_pop(1);
        pthread_cleanup_pop(1);

        return (void*) 0;
}

static void* run_patch_sender(void* q)
{
        struct nv2_label_reader* r = (struct nv2_label_reader*) q;

        int show_warnings = 1;
        int nfail = 0;

        while (1)
        {
                struct nv2_image_patch p;

                pthread_mutex_lock(&r->m_patch_lock);
                p = r->m_patch;
                nv2_image_patch_init_empty(&r->m_patch);
                pthread_mutex_unlock(&r->m_patch_lock);

                if (p.data == 0 || p.width == 0 || p.height == 0)
                {
                        // there was no valid pending patch, so let's
                        // wait very briefly and then check again
                        nv2_image_patch_destroy(&p);
                        usleep(1000);

                        continue;
                }

                r->m_latest_patch_send_result =
                        nv2_label_reader_send_patch_sync
                        (r, &p, show_warnings);

                if (NV2_PATCH_SEND_FAIL
                    == r->m_latest_patch_send_result)
                {
                        sleep(1);

                        // ok, we've already showed one warning about
                        // the patch send failing so let's not show
                        // any more consecutive warnings
                        show_warnings = 0;

                        // keep track of how many consecutive failures
                        // we've had so that we can print a warning
                        // about that if/when we finally do a
                        // successful send again
                        ++nfail;
                }
                else
                {
                        show_warnings = 1;

                        if (nfail > 0)
                        {
                                char buf[64];
                                snprintf(&buf[0], sizeof(buf),
                                         "%d consecutive patch "
                                         "send attempts failed",
                                         nfail);
                                errno = 0;
                                nv2_warn(&buf[0]);
                        }
                        nfail = 0;
                }
        }

        return (void*) 0;
}

struct nv2_label_reader* nv2_label_reader_create(
        const int label_reader_port,
        const char* remote_patch_reader_addr,
        const int remote_patch_reader_port)
{
        struct nv2_label_reader* p =
                malloc(sizeof(struct nv2_label_reader));

        if (p == 0)
                nv2_fatal("malloc() failed");

        pthread_mutex_init(&p->m_label_lock, 0);
        nv2_patch_label_init_empty(&p->m_label);

        pthread_mutex_init(&p->m_patch_lock, 0);
        nv2_image_patch_init_empty(&p->m_patch);
        p->m_latest_patch_send_result = NV2_PATCH_SEND_OK;

        p->m_label_reader_port = label_reader_port;

        if (inet_aton(remote_patch_reader_addr,
                      &p->m_remote_patch_reader_in_addr) == 0)
                nv2_fatal("inet_aton() failed");

        p->m_remote_patch_reader_port = remote_patch_reader_port;

        p->m_reader_socket_fd = -1;

        if (0 != pthread_create(&p->m_label_reader_thread, 0,
                                &run_label_reader,
                                (void*) p))
                nv2_fatal("pthread_create() failed (label reader thread)");

        if (0 != pthread_create(&p->m_patch_sender_thread, 0,
                                &run_patch_sender,
                                (void*) p))
                nv2_fatal("pthread_create() failed (patch sender thread)");

        return p;
}

void nv2_label_reader_destroy(struct nv2_label_reader* p)
{
        pthread_cancel(p->m_label_reader_thread);
        pthread_cancel(p->m_patch_sender_thread);

        if (p->m_reader_socket_fd > 0)
        {
                const int close_me = p->m_reader_socket_fd;
                p->m_reader_socket_fd = -1;
                close(close_me);
        }

        if (0 != pthread_join(p->m_label_reader_thread, 0))
                nv2_fatal("pthread_join() failed");

        if (0 != pthread_join(p->m_patch_sender_thread, 0))
                nv2_fatal("pthread_join() failed");

        pthread_mutex_destroy(&p->m_label_lock);
        pthread_mutex_destroy(&p->m_patch_lock);
        nv2_image_patch_destroy(&p->m_patch);
}

int nv2_label_reader_get_current_label(struct nv2_label_reader* p,
                                       struct nv2_patch_label* l)
{
        pthread_mutex_lock(&p->m_label_lock);
        *l = p->m_label;
        nv2_patch_label_init_empty(&p->m_label);
        pthread_mutex_unlock(&p->m_label_lock);

        if (l->patch_id == 0)
                return 0;
        else
                return 1;
}

void
nv2_label_reader_send_patch(struct nv2_label_reader* r,
                            struct nv2_image_patch* p)
{
        pthread_mutex_lock(&r->m_patch_lock);
        nv2_image_patch_destroy(&r->m_patch);
        r->m_patch = *p;
        nv2_image_patch_init_empty(p);
        pthread_mutex_unlock(&r->m_patch_lock);
}

enum nv2_patch_send_result
nv2_label_reader_send_patch_sync(struct nv2_label_reader* r,
                                 struct nv2_image_patch* p,
                                 const int show_warnings)
{
        const int socket_fd = socket(PF_INET, SOCK_STREAM, 0);

        if (socket_fd == -1)
                nv2_fatal("socket() failed");

        struct sockaddr_in name;
        name.sin_family = AF_INET;
        name.sin_addr = r->m_remote_patch_reader_in_addr;
        name.sin_port = htons(r->m_remote_patch_reader_port);

        struct nv2_image_patch np; // network-byte-order patch

        if (p) np = *p;
        else nv2_image_patch_init_empty(&np);

        const uint32_t npixbytes =
                np.width * np.height * nv2_pixel_type_bytes_per_pixel(np.type);

        np.protocol_version  = htonl(np.protocol_version);
        np.width             = htonl(np.width);
        np.height            = htonl(np.height);
        np.id                = htonl(np.id);
        np.is_training_image = htonl(np.is_training_image);
        np.type              = htonl(np.type);
        np.training_label[sizeof(np.training_label)-1] = '\0';
        np.remote_command[sizeof(np.remote_command)-1] = '\0';

        enum nv2_patch_send_result retval = NV2_PATCH_SEND_OK;

        if (connect(socket_fd, (struct sockaddr*)(&name), sizeof(name))
            == -1)
        {
                if (show_warnings)
                        nv2_warn("while attempting to send patch: "
                                 "connect() failed");
                retval = NV2_PATCH_SEND_FAIL;
        }
        else if (send(socket_fd, &np, NV2_IMAGE_PATCH_HEADER_SIZE, 0)
                 != NV2_IMAGE_PATCH_HEADER_SIZE)
        {
                if (show_warnings)
                        nv2_warn("while attempting to send patch: "
                                 "send(header) failed");
                retval = NV2_PATCH_SEND_FAIL;
        }
        else if (npixbytes > 0)
        {
                const ssize_t nwritten =
                        nv2_robust_write(socket_fd, np.data, npixbytes);

                if (nwritten != npixbytes)
                {
                        if (show_warnings)
                                nv2_warn("while attempting to send patch: "
                                         "send(pixels) failed");
                        retval = NV2_PATCH_SEND_FAIL;
                }
        }

        close(socket_fd);

        nv2_image_patch_destroy(p);

        return retval;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* c-file-style: "linux" */
/* End: */

#endif // NEOVISIONII_NV2_LABEL_READER_C_DEFINED
