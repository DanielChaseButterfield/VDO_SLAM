#include<string>
#include<Exceptions.h>
using namespace std;

NotImplementedException::NotImplementedException(const char* error) {
    msg = error;
}

const char* NotImplementedException::what() const noexcept {
    return msg.c_str();
}