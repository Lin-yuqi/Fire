#include <Fire/base/base.h>

// ----------------base begin-------------
namespace base{


Status::Status(StatusCode code, std::string message)
    : _code(code), _message(std::move(message)) {}

bool Status::ok() const {
    return _code == StatusCode::Success;
}

Status::operator bool() const {
    return ok();
}

StatusCode Status::code() const {
    return _code;
}

const std::string& Status::message() const {
    return _message;
}

void Status::set_code(StatusCode code) {
    _code = code;
}

void Status::set_message(const std::string& message) {
    _message = message;
}

// ---------------errro begin---------------
namespace error {
Status Success(const std::string& err_msg){
    return Status(StatusCode::Success,err_msg);
}

Status FunctionNotImplement(const std::string& err_msg){
    return Status(StatusCode::FunctionUnImplement,err_msg);
}

Status PathNotValid(const std::string& err_msg){
    return Status(StatusCode::PathNotValid,err_msg);
}

Status ModelParseError(const std::string& err_msg){
    return Status(StatusCode::ModelParseError,err_msg);
}

Status InternalError(const std::string& err_msg){
    return Status(StatusCode::InternalError,err_msg);
}

Status KeyHasExits(const std::string& err_msg){
    return Status(StatusCode::KeyValueHasExist,err_msg);
}

Status InvalidArgument(const std::string& err_msg){
    return Status(StatusCode::InvalidArgument,err_msg);
}

}
//----------------error end-----------------

}
// ----------------base end--------------