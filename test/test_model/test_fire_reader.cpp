#include "Fire/model/fire_reader.h"
#include "Fire/tensor/tensor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {

constexpr size_t kHeaderSize = 32;
constexpr size_t kTensorInfoSize = 96;
constexpr size_t kDataOffset = 224;
constexpr size_t kNormOffset = 224;
constexpr size_t kAttentionWqOffset = 236;
constexpr size_t kFixtureSize = 260;

uint32_t ReadU32Le(const uint8_t* bytes) {
    return static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
           (static_cast<uint32_t>(bytes[2]) << 16) |
           (static_cast<uint32_t>(bytes[3]) << 24);
}

uint64_t ReadU64Le(const uint8_t* bytes) {
    return static_cast<uint64_t>(bytes[0]) | (static_cast<uint64_t>(bytes[1]) << 8) |
           (static_cast<uint64_t>(bytes[2]) << 16) |
           (static_cast<uint64_t>(bytes[3]) << 24) |
           (static_cast<uint64_t>(bytes[4]) << 32) |
           (static_cast<uint64_t>(bytes[5]) << 40) |
           (static_cast<uint64_t>(bytes[6]) << 48) |
           (static_cast<uint64_t>(bytes[7]) << 56);
}

void WriteU32Le(std::vector<uint8_t>& bytes, size_t offset, uint32_t value) {
    for (size_t index = 0; index < 4; ++index) {
        bytes.at(offset + index) = static_cast<uint8_t>(value >> (index * 8));
    }
}

void WriteU64Le(std::vector<uint8_t>& bytes, size_t offset, uint64_t value) {
    for (size_t index = 0; index < 8; ++index) {
        bytes.at(offset + index) = static_cast<uint8_t>(value >> (index * 8));
    }
}

std::vector<uint8_t> ReadFixture() {
    std::ifstream input(FIRE_V1_FIXTURE_PATH, std::ios::binary);
    if (!input) {
        throw std::runtime_error(std::string("failed to open fixture: ") + FIRE_V1_FIXTURE_PATH);
    }
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(input),
                                std::istreambuf_iterator<char>());
}

class TemporaryFireFile final {
  public:
    explicit TemporaryFireFile(const std::vector<uint8_t>& bytes) {
        std::array<char, 40> path_template{};
        const char pattern[] = "/tmp/fire_reader_test_XXXXXX";
        std::copy(std::begin(pattern), std::end(pattern), path_template.begin());
        const int descriptor = ::mkstemp(path_template.data());
        if (descriptor < 0) {
            throw std::runtime_error("mkstemp failed for FireReader test");
        }
        ::close(descriptor);
        _path = path_template.data();

        std::ofstream output(_path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
        if (!output) {
            throw std::runtime_error("failed to write temporary FireReader test file");
        }
    }

    ~TemporaryFireFile() { std::remove(_path.c_str()); }

    TemporaryFireFile(const TemporaryFireFile&) = delete;
    TemporaryFireFile& operator=(const TemporaryFireFile&) = delete;

    const std::string& path() const { return _path; }

  private:
    std::string _path;
};

void ExpectModelParseError(const std::string& label, const std::vector<uint8_t>& bytes) {
    TemporaryFireFile file(bytes);
    model::FireReader reader;

    const base::Status status = reader.open(file.path());

    EXPECT_EQ(status.code(), base::StatusCode::ModelParseError)
        << label << ": " << status.message();
    EXPECT_EQ(reader.find("norm.weight"), nullptr) << label;
    EXPECT_EQ(reader.mapped_buffer(), nullptr) << label;
}

} // namespace

TEST(FireReaderTest, UnopenedReaderHasNoTensorOrMappedBuffer) {
    model::FireReader reader;

    EXPECT_EQ(reader.find("norm.weight"), nullptr);
    EXPECT_EQ(reader.mapped_buffer(), nullptr);
}

TEST(FireReaderTest, OpensTheIndependentWireContractFixture) {
    model::FireReader reader;
    const base::Status status = reader.open(FIRE_V1_FIXTURE_PATH);
    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_EQ(status.code(), base::StatusCode::Success);

    const std::shared_ptr<base::Buffer> mapping = reader.mapped_buffer();
    ASSERT_NE(mapping, nullptr);
    ASSERT_EQ(mapping->size(), kFixtureSize);
    EXPECT_EQ(mapping->device_type(), base::DeviceType::CPU);

    const auto* wire = static_cast<const uint8_t*>(mapping->ptr());
    ASSERT_NE(wire, nullptr);
    EXPECT_EQ(std::memcmp(wire, "FIRECKPT", 8), 0);
    EXPECT_EQ(ReadU32Le(wire + 8), 1U);
    EXPECT_EQ(ReadU32Le(wire + 12), 2U);
    EXPECT_EQ(ReadU64Le(wire + 16), kHeaderSize);
    EXPECT_EQ(ReadU64Le(wire + 24), kDataOffset);

    EXPECT_EQ(ReadU64Le(wire + kHeaderSize + 64), kNormOffset);
    EXPECT_EQ(ReadU64Le(wire + kHeaderSize + 72), 12U);
    EXPECT_EQ(ReadU32Le(wire + kHeaderSize + 80), 3U);
    EXPECT_EQ(ReadU32Le(wire + kHeaderSize + 84), 0U);
    EXPECT_EQ(wire[kHeaderSize + 88], 1U);
    EXPECT_EQ(wire[kHeaderSize + 89], 1U);

    const size_t second_record = kHeaderSize + kTensorInfoSize;
    EXPECT_EQ(ReadU64Le(wire + second_record + 64), kAttentionWqOffset);
    EXPECT_EQ(ReadU64Le(wire + second_record + 72), 24U);
    EXPECT_EQ(ReadU32Le(wire + second_record + 80), 2U);
    EXPECT_EQ(ReadU32Le(wire + second_record + 84), 3U);
    EXPECT_EQ(wire[second_record + 88], 1U);
    EXPECT_EQ(wire[second_record + 89], 2U);

    const model::TensorInfo* norm = reader.find("norm.weight");
    ASSERT_NE(norm, nullptr);
    EXPECT_EQ(norm->name, "norm.weight");
    EXPECT_EQ(norm->dtype, base::DataType::Fp32);
    ASSERT_EQ(norm->dims.size(), 1U);
    EXPECT_EQ(norm->dims[0], 3);
    EXPECT_EQ(norm->byte_offset, kNormOffset);
    EXPECT_EQ(norm->byte_size, 12U);

    const model::TensorInfo* attention = reader.find("layers.0.attention.wq.weight");
    ASSERT_NE(attention, nullptr);
    EXPECT_EQ(attention->dtype, base::DataType::Fp32);
    ASSERT_EQ(attention->dims.size(), 2U);
    EXPECT_EQ(attention->dims[0], 2);
    EXPECT_EQ(attention->dims[1], 3);
    EXPECT_EQ(attention->byte_offset, kAttentionWqOffset);
    EXPECT_EQ(attention->byte_size, 24U);
    EXPECT_EQ(reader.find("missing.weight"), nullptr);

    EXPECT_EQ(ReadU32Le(wire + kNormOffset), 0x3f800000U);
    EXPECT_EQ(ReadU32Le(wire + kNormOffset + 4), 0xc0000000U);
    EXPECT_EQ(ReadU32Le(wire + kNormOffset + 8), 0x3fc00000U);

    const auto* matrix = reinterpret_cast<const float*>(wire + kAttentionWqOffset);
    const std::array<float, 6> expected = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    for (size_t row = 0; row < 2; ++row) {
        for (size_t column = 0; column < 3; ++column) {
            EXPECT_FLOAT_EQ(matrix[row * 3 + column], expected[row * 3 + column]);
        }
    }
}

TEST(FireReaderTest, RejectsMalformedHeaderDirectoryAndPayload) {
    using Mutation = std::function<void(std::vector<uint8_t>&)>;
    const std::vector<std::pair<std::string, Mutation>> cases = {
        {"bad magic", [](auto& bytes) { bytes[0] ^= 0xffU; }},
        {"bad version", [](auto& bytes) { WriteU32Le(bytes, 8, 2); }},
        {"wrong directory offset", [](auto& bytes) { WriteU64Le(bytes, 16, 31); }},
        {"wrong data offset", [](auto& bytes) { WriteU64Le(bytes, 24, 225); }},
        {"truncated header", [](auto& bytes) { bytes.resize(kHeaderSize - 1); }},
        {"truncated directory", [](auto& bytes) { bytes.resize(kDataOffset - 1); }},
        {"truncated payload", [](auto& bytes) { bytes.resize(kFixtureSize - 1); }},
        {"empty name", [](auto& bytes) {
             std::fill(bytes.begin() + 32, bytes.begin() + 96, 0);
         }},
        {"unterminated name", [](auto& bytes) {
             std::fill(bytes.begin() + 32, bytes.begin() + 96, 'a');
         }},
        {"nonzero after name terminator", [](auto& bytes) { bytes[44] = 'x'; }},
        {"non-ASCII name", [](auto& bytes) { bytes[32] = 0x80U; }},
        {"duplicate name", [](auto& bytes) {
             std::copy(bytes.begin() + 32, bytes.begin() + 96, bytes.begin() + 128);
         }},
        {"nonzero TensorInfo padding", [](auto& bytes) { bytes[122] = 1; }},
        {"rank-1 unused shape", [](auto& bytes) { WriteU32Le(bytes, 116, 1); }},
        {"wrong dtype", [](auto& bytes) { bytes[120] = 2; }},
        {"wrong ndim", [](auto& bytes) { bytes[121] = 3; }},
        {"zero shape", [](auto& bytes) { WriteU32Le(bytes, 112, 0); }},
        {"shape outside int32", [](auto& bytes) { WriteU32Le(bytes, 112, 0x80000000U); }},
        {"shape-byte-size mismatch", [](auto& bytes) { WriteU64Le(bytes, 104, 8); }},
        {"unaligned payload", [](auto& bytes) { WriteU64Le(bytes, 96, 225); }},
        {"non-contiguous payload", [](auto& bytes) { WriteU64Le(bytes, 192, 240); }},
        {"payload out of bounds", [](auto& bytes) {
             WriteU64Le(bytes, 200, 800);
             WriteU32Le(bytes, 212, 100);
         }},
        {"near uint64 byte size", [](auto& bytes) {
             WriteU64Le(bytes, 104, std::numeric_limits<uint64_t>::max());
         }},
        {"near uint64 data offset", [](auto& bytes) {
             WriteU64Le(bytes, 24, std::numeric_limits<uint64_t>::max());
         }},
        {"huge directory", [](auto& bytes) {
             WriteU32Le(bytes, 12, std::numeric_limits<uint32_t>::max());
             WriteU64Le(bytes, 24,
                        32ULL + static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) *
                                    kTensorInfoSize);
         }},
        {"file tail mismatch", [](auto& bytes) { bytes.push_back(0); }},
    };

    const std::vector<uint8_t> valid = ReadFixture();
    ASSERT_EQ(valid.size(), kFixtureSize);
    for (const auto& [label, mutate] : cases) {
        SCOPED_TRACE(label);
        std::vector<uint8_t> malformed = valid;
        mutate(malformed);
        ExpectModelParseError(label, malformed);
    }
}

TEST(FireReaderTest, RejectsPayloadRangeArithmeticOverflow) {
    std::vector<uint8_t> malformed = ReadFixture();
    WriteU64Le(malformed, 96, std::numeric_limits<uint64_t>::max() - 3);
    TemporaryFireFile file(malformed);
    model::FireReader reader;

    const base::Status status = reader.open(file.path());

    EXPECT_EQ(status.code(), base::StatusCode::ModelParseError);
    EXPECT_NE(status.message().find("payload range overflow"), std::string::npos)
        << status.message();
    EXPECT_EQ(reader.find("norm.weight"), nullptr);
    EXPECT_EQ(reader.mapped_buffer(), nullptr);
}

TEST(FireReaderTest, ReportsRecoverableStatusForBadExternalFiles) {
    model::FireReader empty_path_reader;
    const base::Status empty_path = empty_path_reader.open("");
    EXPECT_EQ(empty_path.code(), base::StatusCode::PathNotValid);
    EXPECT_EQ(empty_path_reader.find("norm.weight"), nullptr);
    EXPECT_EQ(empty_path_reader.mapped_buffer(), nullptr);

    const std::string missing_path =
        "/tmp/fire_reader_missing_" + std::to_string(::getpid()) + ".fire";
    std::remove(missing_path.c_str());
    model::FireReader missing_path_reader;
    const base::Status missing = missing_path_reader.open(missing_path);
    EXPECT_EQ(missing.code(), base::StatusCode::PathNotValid);
    EXPECT_EQ(missing_path_reader.find("norm.weight"), nullptr);
    EXPECT_EQ(missing_path_reader.mapped_buffer(), nullptr);

    model::FireReader malformed_reader;
    TemporaryFireFile malformed(std::vector<uint8_t>(kHeaderSize - 1, 0));
    const base::Status malformed_status = malformed_reader.open(malformed.path());
    EXPECT_EQ(malformed_status.code(), base::StatusCode::ModelParseError);
    EXPECT_EQ(malformed_reader.find("norm.weight"), nullptr);
    EXPECT_EQ(malformed_reader.mapped_buffer(), nullptr);

    model::FireReader system_error_reader;
    const base::Status system_error =
        system_error_reader.open(std::filesystem::temp_directory_path().string());
    EXPECT_EQ(system_error.code(), base::StatusCode::InternalError) << system_error.message();
    EXPECT_EQ(system_error_reader.find("norm.weight"), nullptr);
    EXPECT_EQ(system_error_reader.mapped_buffer(), nullptr);
}

TEST(FireReaderTest, TensorKeepsMappingAliveAfterReaderDestruction) {
    std::unique_ptr<tensor::Tensor> tensor_view;
    {
        model::FireReader reader;
        const base::Status status = reader.open(FIRE_V1_FIXTURE_PATH);
        ASSERT_TRUE(status.ok()) << status.message();
        const model::TensorInfo* info = reader.find("layers.0.attention.wq.weight");
        ASSERT_NE(info, nullptr);
        tensor_view = std::make_unique<tensor::Tensor>(
            info->dtype, info->dims, reader.mapped_buffer(), static_cast<size_t>(info->byte_offset));
    }

    ASSERT_NE(tensor_view, nullptr);
    ASSERT_NE(tensor_view->ptr<float>(), nullptr);
    EXPECT_EQ(tensor_view->dims_size(), 2);
    EXPECT_EQ(tensor_view->get_dim(0), 2);
    EXPECT_EQ(tensor_view->get_dim(1), 3);
    const std::array<float, 6> expected = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    for (size_t index = 0; index < expected.size(); ++index) {
        EXPECT_FLOAT_EQ(tensor_view->ptr<float>()[index], expected[index]);
    }
}
