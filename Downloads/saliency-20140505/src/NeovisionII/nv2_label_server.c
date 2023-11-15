/*!@file NeovisionII/nv2_label_server.c */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/nv2_label_server.c $
// $Id: nv2_label_server.c 9765 2008-05-12 16:58:24Z rjpeters $
//

#ifndef NEOVISIONII_NV2_LABEL_SERVER_C_DEFINED
#define NEOVISIONII_NV2_LABEL_SERVER_C_DEFINED

#include "NeovisionII/nv2_label_server.h"

#include "NeovisionII/nv2_common.h"

// #include order matters here:
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct nv2_label_server
{
        pthread_mutex_t m_patch_lock;
        struct nv2_image_patch m_patch;
        int m_should_quit;

        int m_patch_reader_port;
        struct in_addr m_remote_label_reader_in_addr;
        int m_remote_label_reader_port;

        int m_verbosity;

        volatile int socket_fd;

        pthread_t m_patch_reader_thread;
};

static void* run_patch_reader(void* vq)
{
        struct nv2_label_server* const q = (struct nv2_label_server*) vq;

        q->socket_fd = socket(PF_INET, SOCK_STREAM, 0);
        if (q->socket_fd == -1)
                nv2_fatal("socket() failed");

        {
                const int set_on = 1;
                if (setsockopt(q->socket_fd,
                               SOL_SOCKET, SO_REUSEADDR,
                               &set_on, sizeof(set_on)) == -1)
                        nv2_warn("setsockopt() failed");
        }

        struct sockaddr_in name;
        name.sin_family = AF_INET;
        name.sin_addr.s_addr = htonl(INADDR_ANY); // accept from any host
        name.sin_port = htons(q->m_patch_reader_port);
        if (bind(q->socket_fd, (struct sockaddr*)(&name), sizeof(name)) == -1)
                nv2_fatal("bind() failed");

        listen(q->socket_fd, 5);

        int client_socket_fd = -1;

        // Accept incoming connections
        while (q->socket_fd > 0)
        {
                struct sockaddr_in client_name;
                socklen_t client_name_len = sizeof(client_name);

                client_socket_fd =
                        accept(q->socket_fd,
                               (struct sockaddr*)(&client_name),
                               &client_name_len);

                if (client_socket_fd < 0)
                {
                        // if accept() failed because q->socket_fd was
                        // closed() and/or is less than 0, then that
                        // is because the main thread closed the
                        // socket to force us to shut down, so let's
                        // quit without any nv2_warn()
                        if (q->socket_fd < 0)
                                break;

                        if (q->m_verbosity >= 1)
                                fprintf(stderr, "accept() failed "
                                        "(strerror=%s)\n",
                                        strerror(errno));
                        continue;
                }

                struct nv2_image_patch p;
                nv2_image_patch_init_empty(&p);

                if (nv2_robust_read(client_socket_fd, &p,
                                    NV2_IMAGE_PATCH_HEADER_SIZE, NULL)
                    != NV2_IMAGE_PATCH_HEADER_SIZE)
                {
                        if (q->m_verbosity >= 1)
                                fprintf(stderr, "read(header) failed\n");
                        break;
                }

                p.protocol_version = ntohl(p.protocol_version);
                p.width            = ntohl(p.width);
                p.height           = ntohl(p.height);
                p.id               = ntohl(p.id);
                p.is_training_image= ntohl(p.is_training_image);
                p.type             = ntohl(p.type);
                p.training_label[sizeof(p.training_label)-1] = '\0';
                p.remote_command[sizeof(p.remote_command)-1] = '\0';

                const uint32_t npixbytes =
                        p.width * p.height * nv2_pixel_type_bytes_per_pixel(p.type);

                // set a maximum image size of 16MB, so that we avoid
                // trying to allocate too much memory when filling in
                // the pixel array
                const uint32_t maxbytes = 4096*4096;

                int doquit = 0;

                if (npixbytes > maxbytes)
                {
                        if (q->m_verbosity >= 1)
                                fprintf(stderr,
                                        "image patch was %ux%u%u=%u bytes, "
                                        "but the maximum allowable size "
                                        "is %u bytes\n",
                                        (unsigned int) p.width,
                                        (unsigned int) p.height,
                                        (unsigned int) nv2_pixel_type_bytes_per_pixel(p.type),
                                        (unsigned int) npixbytes,
                                        (unsigned int) maxbytes);

                        doquit = 1;
                }
                else if (p.protocol_version == 0)
                {
                        // client will send protocol_version == 0 to
                        // tell us to quit:
                        doquit = 1;
                }
                else if (p.protocol_version != NV2_PATCH_PROTOCOL_VERSION)
                {
                        if (q->m_verbosity >= 1)
                                fprintf(stderr,
                                        "wrong patch protocol version "
                                        "(got %u, expected %u)",
                                        (unsigned int) p.protocol_version,
                                        (unsigned int) NV2_PATCH_PROTOCOL_VERSION);

                        doquit = 1;
                }
                else
                {
                        p.data = (unsigned char*) malloc(npixbytes);

                        int nchunks_read = 0;

                        const size_t n =
                                nv2_robust_read(client_socket_fd,
                                                p.data, npixbytes,
                                                &nchunks_read);

                        if (n != npixbytes)
                        {
                                doquit = 1;
                                if (q->m_verbosity >= 1)
                                        nv2_warn("read(pixels) failed");
                        }

                        if (q->m_verbosity >= 2)
                                fprintf(stderr,
                                        "got patch = { .protocol_version=%u, "
                                        ".width=%u, .height=%u, .id=%u, "
                                        ".is_training=%u, "
                                        ".label='%s', .cmd='%s', "
                                        ".data=[%d bytes in %d chunks] }\n",
                                        (unsigned int) p.protocol_version,
                                        (unsigned int) p.width,
                                        (unsigned int) p.height,
                                        (unsigned int) p.id,
                                        (unsigned int) p.is_training_image,
                                        &p.training_label[0],
                                        &p.remote_command[0],
                                        (int) n, (int) nchunks_read);
                }

                close(client_socket_fd);
                client_socket_fd = -1;

                if (doquit)
                        break;

                pthread_mutex_lock(&q->m_patch_lock);
                nv2_image_patch_destroy(&q->m_patch);
                q->m_patch = p;
                pthread_mutex_unlock(&q->m_patch_lock);
        }

        pthread_mutex_lock(&q->m_patch_lock);
        q->m_should_quit = 1;
        pthread_mutex_unlock(&q->m_patch_lock);

        if (client_socket_fd >= 0)
                close(client_socket_fd);

        close(q->socket_fd);

        return (void*) 0;
}

struct nv2_label_server* nv2_label_server_create(
        const int patch_reader_port,
        const char* remote_label_reader_addr,
        const int remote_label_reader_port)
{
        struct nv2_label_server* p =
                (struct nv2_label_server*)
                malloc(sizeof(struct nv2_label_server));

        if (p == 0)
                nv2_fatal("malloc() failed");

        pthread_mutex_init(&p->m_patch_lock, 0);
        nv2_image_patch_init_empty(&p->m_patch);
        p->m_should_quit = 0;

        p->m_patch_reader_port = patch_reader_port;

        if (inet_aton(remote_label_reader_addr,
                      &p->m_remote_label_reader_in_addr) == 0)
                nv2_fatal("inet_aton() failed");

        p->m_remote_label_reader_port = remote_label_reader_port;

        p->socket_fd = -1;

        p->m_verbosity = 2;

        if (0 != pthread_create(&p->m_patch_reader_thread, 0,
                                &run_patch_reader,
                                (void*) p))
                nv2_fatal("pthread_create() failed");

        return p;
}


void nv2_label_server_destroy(struct nv2_label_server* p)
{
        if (p->socket_fd != -1)
        {
                const int oldfd = p->socket_fd;
                p->socket_fd = -1;
                close(oldfd);
        }

        if (0 != pthread_join(p->m_patch_reader_thread, 0))
                nv2_fatal("pthread_join() failed");
        nv2_image_patch_destroy(&p->m_patch);
        pthread_mutex_destroy(&p->m_patch_lock);
}

enum nv2_image_patch_result
nv2_label_server_get_current_patch(struct nv2_label_server* p,
                                   struct nv2_image_patch* ret)
{
        pthread_mutex_lock(&p->m_patch_lock);
        *ret = p->m_patch;
        const int doquit = p->m_should_quit;
        nv2_image_patch_init_empty(&p->m_patch);
        pthread_mutex_unlock(&p->m_patch_lock);

        if (doquit)             return NV2_IMAGE_PATCH_END;
        else if (ret->id > 0)   return NV2_IMAGE_PATCH_VALID;
        else                    return NV2_IMAGE_PATCH_NONE;
}

enum nv2_label_send_result
nv2_label_server_send_label(struct nv2_label_server* p,
                            const struct nv2_patch_label* l)
{
        const int socket_fd = socket(PF_INET, SOCK_STREAM, 0);

        if (socket_fd == -1)
                nv2_fatal("socket() failed");

        // Specify the server address and port
        struct sockaddr_in name;
        name.sin_family = AF_INET;
        name.sin_addr = p->m_remote_label_reader_in_addr;
        name.sin_port = htons(p->m_remote_label_reader_port);

        struct nv2_patch_label nl = *l;
        nl.protocol_version = htonl(l->protocol_version);
        nl.patch_id = htonl(l->patch_id);
        nl.confidence = htonl(l->confidence);

        enum nv2_label_send_result retval = NV2_LABEL_SEND_OK;

        errno = 0;
        if (connect(socket_fd, (struct sockaddr*)(&name), sizeof(name))
            == -1)
        {
                if (p->m_verbosity >= 1)
                        nv2_warn("connect() failed");
                retval = NV2_LABEL_SEND_FAIL;
        }
        else if (nv2_robust_write(socket_fd, &nl, sizeof(nl))
                 != sizeof(nl))
        {
                if (p->m_verbosity >= 1)
                        nv2_warn("write(nv2_patch_label) failed");
                retval = NV2_LABEL_SEND_FAIL;
        }

        close(socket_fd);

        return retval;
}

void nv2_label_server_set_verbosity(struct nv2_label_server* p,
                                    const int verbosity)
{
        p->m_verbosity = verbosity;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* c-file-style: "linux" */
/* End: */

#endif // NEOVISIONII_NV2_LABEL_SERVER_C_DEFINED
