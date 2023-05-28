

#include <iostream>
#include <string>
#include <map>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

using namespace std;

int main()
{
    int sockfd, newsockfd, portno;
    socklen_t clilen;
    char buffer[256];
    struct sockaddr_in serv_addr, cli_addr;
    int n;

    // Create a socket
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) 
        cout << "ERROR opening socket" << endl;

    // Set all values in buffer to 0
    bzero((char *) &serv_addr, sizeof(serv_addr));

    // Set port number
    portno = 8080;

    // Set up the socket
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);

    // Bind the socket
    if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) 
        cout << "ERROR on binding" << endl;

    // Listen for connections
    listen(sockfd, 5);
    clilen = sizeof(cli_addr);

    // Accept a connection
    newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
    if (newsockfd < 0) 
        cout << "ERROR on accept" << endl;

    // Read the request
    bzero(buffer, 256);
    n = read(newsockfd, buffer, 255);
    if (n < 0) 
        cout << "ERROR reading from socket" << endl;

    // Parse the request
    string request(buffer);
    string method = request.substr(0, request.find(' '));
    string uri = request.substr(request.find(' ')+1, request.find(' ', request.find(' ')+1)-request.find(' ')-1);

    // Create response
    string response;
    if (method == "GET" && uri == "/derp") {
        map<string, string> response_headers;
        response_headers["Content-Type"] = "application/json";
        response_headers["Connection"] = "close";

        string body = "{\"message\": \"hello nerd\"}";

        response = "HTTP/1.1 200 OK\r\n";
        for (map<string, string>::iterator it = response_headers.begin(); it != response_headers.end(); it++) {
            response += it->first + ": " + it->second + "\r\n";
        }
        response += "Content-Length: " + to_string(body.length()) + "\r\n\r\n";
        response += body;
    } else {
        response = "HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n";
    }

    // Send response
    n = write(newsockfd, response.c_str(), response.length());
    if (n < 0) 
        cout << "ERROR writing to socket" << endl;

    close(newsockfd);
    close(sockfd);

    return 0;
}