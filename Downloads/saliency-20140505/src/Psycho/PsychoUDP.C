#include "Psycho/PsychoUDP.H"
#include <unistd.h>

PsychoUDP::PsychoUDP(const char *node, const char *port, int buf_size) :
  bufSize(uint(buf_size)),
  buf(new char[buf_size])
{
  memset(&hints, 0, sizeof(struct addrinfo));
  hints.ai_family = AF_UNSPEC;    /* Allow IPv4 or IPv6 */
  hints.ai_socktype = SOCK_DGRAM; /* Datagram socket */
  hints.ai_flags = 0;
  hints.ai_protocol = 0;          /* Any protocol */
  
  s = getaddrinfo(node, port, &hints, &result);
  if (s != 0) LFATAL("getaddrinfo: %s", gai_strerror(s));


  /* getaddrinfo() returns a list of address structures.
     Try each address until we successfully connect(2).
     If socket(2) (or connect(2)) fails, we (close the socket
     and) try the next address. */

  for (rp = result; rp != NULL; rp = rp->ai_next) {
    sfd = socket(rp->ai_family, rp->ai_socktype,
		 rp->ai_protocol);
    if (sfd == -1)
      continue;

    if (connect(sfd, rp->ai_addr, rp->ai_addrlen) != -1)
      break;                  /* Success */

    close(sfd);
  }

  if (rp == NULL) {               /* No address succeeded */
    LFATAL("Could not connect %s:%s",node,port);
  }

  freeaddrinfo(result);           /* No longer needed */
}

bool PsychoUDP::sendData(std::string data)
{
  LINFO("Data send:%s",data.c_str());
  size_t len = strlen(data.c_str()) + 1; //+1 for terminating null byte
  
  if (len + 1 > bufSize ){
    LERROR("Message is too long");
    return false;
  }

  if (write(sfd, data.c_str(), len) != int(len)) {
    LERROR("partial/failed write");
    return false;
  }

  return true;
  //nread = read(sfd, buf, BUF_SIZE);
  //if (nread == -1) {
  //perror("read");
  //exit(EXIT_FAILURE);
  //}
  //printf("Received %ld bytes: %s\n", (long) nread, buf);
}


