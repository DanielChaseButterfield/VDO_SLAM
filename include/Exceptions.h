#include<string>
using namespace std;

class NotImplementedException : public std::exception {
    public:
        // Construct the exception with the default message
        NotImplementedException(const char* error);

        // For compatibility with std::exception
        const char* what() const noexcept;
        
    private:
        std::string msg;
};