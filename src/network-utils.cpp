#include "network-utils.h"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

bool is_port_open(const std::string& ip, uint32_t port, int timeout_sec) {
#ifdef _WIN32
    static bool wsa_init = false;
    if (!wsa_init) { WSADATA wsa; WSAStartup(MAKEWORD(2, 2), &wsa); wsa_init = true; }
#endif

    int sock = (int) socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return false;

#ifdef _WIN32
    DWORD tv = (DWORD) timeout_sec * 1000; // SO_*TIMEO is milliseconds on Windows
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof(tv));
#else
    struct timeval tv;
    tv.tv_sec = timeout_sec;
    tv.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif

    struct sockaddr_in server;
    server.sin_addr.s_addr = inet_addr(ip.c_str());
    server.sin_family = AF_INET;
    server.sin_port = htons(port);

    int res = connect(sock, (struct sockaddr*)&server, sizeof(server));
#ifdef _WIN32
    closesocket(sock);
#else
    close(sock);
#endif
    return res == 0;
}
