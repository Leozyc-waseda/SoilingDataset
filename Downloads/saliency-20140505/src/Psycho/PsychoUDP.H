#include "Util/log.H"
//#include "Component/ModelManager.H"
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
//#include <stdio.h>
//#include <stdlib.h>
//#include <unistd.h>
//#include <string.h>


//#define BUF_SIZE 500

#ifndef PSYCHOUDP_H
#define PSYCHOUDP_H

class PsychoUDP
{
public:
  PsychoUDP(const char *node, const char *port, int buf_size = 500);
  ~PsychoUDP();

  bool sendData(std::string data = "");

private:
  struct addrinfo hints;
  struct addrinfo *result, *rp;
  int sfd, s, j;
  uint bufSize;
  ssize_t nread;
  char *buf;

};

#endif
