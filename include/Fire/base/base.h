#pragma once
#include <string>
#include <cstdint>


// ------------------------------base begin---------------------
namespace base{
    
enum class DeviceType:uint8_t{
    Unknown = 0,
    GPU = 1,
    CPU 
};

enum class MemCpyKind{
    CPU2GPU=0,
    CPU2CPU,
    GPU2CPU,
    GPU2GPU
};

class NoCopyable{
protected:
    NoCopyable() = default;
    ~NoCopyable() = default;

    NoCopyable(NoCopyable& x)=delete;
    NoCopyable& operator=(const NoCopyable&) = delete;
};

enum class DataType{
    Unknown=0,
    Fp32,
    int8,
    int32
};

enum StatusCode : uint8_t {
  Success = 0,
  FunctionUnImplement = 1,
  PathNotValid = 2,
  ModelParseError = 3,
  InternalError = 5,
  KeyValueHasExist = 6,
  InvalidArgument = 7,
};

class Status {
public:
    Status() = default;

    Status(StatusCode code,std::string message="");

    bool ok() const;

    explicit operator bool()const;

    StatusCode code()const;

    const std::string& message()const;

    void set_code(StatusCode code);

    void set_message(const std::string& message);

private:
    StatusCode _code = StatusCode::Success;
    std::string _message;
};

namespace error {
#define STATUS_CHECK(call)                                                                 \
  do {                                                                                     \
    const base::Status& status = call;                                                     \
    if (!status) {                                                                         \
      const size_t buf_size = 512;                                                         \
      char buf[buf_size];                                                                  \
      snprintf(buf, buf_size - 1,                                                          \
               "Infer error\n File:%s Line:%d\n Error code:%d\n Error msg:%s\n", __FILE__, \
               __LINE__, int(status), status.get_err_msg().c_str());                       \
      LOG(FATAL) << buf;                                                                   \
    }                                                                                      \
  } while (0)
Status Success(const std::string& err_msg = "");

Status FunctionNotImplement(const std::string& err_msg = "");

Status PathNotValid(const std::string& err_msg = "");

Status ModelParseError(const std::string& err_msg = "");

Status InternalError(const std::string& err_msg = "");

Status KeyHasExits(const std::string& err_msg = "");

Status InvalidArgument(const std::string& err_msg = "");
}

}
// ------------------------------base end---------------------