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



}
// ------------------------------base end---------------------