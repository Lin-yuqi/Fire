#include "Fire/model/fire_reader.h"

#include <gtest/gtest.h>

TEST(FireReaderTest, UnopenedReaderHasNoTensorOrMappedBuffer) {
    model::FireReader reader;

    EXPECT_EQ(reader.find("norm.weight"), nullptr);
    EXPECT_EQ(reader.mapped_buffer(), nullptr);
}

// The remaining tests are acceptance-test placeholders for the implementation
// phase. Enable them one at a time during the red-green loop.
TEST(FireReaderTest, DISABLED_OpensThePythonWireContractFixture) {
    FAIL() << "TODO: validate both tensor records and their FP32 values";
}

TEST(FireReaderTest, DISABLED_RejectsMalformedHeaderDirectoryAndPayload) {
    FAIL() << "TODO: mutate the valid fixture for each documented invariant";
}

TEST(FireReaderTest, DISABLED_ReportsRecoverableStatusForBadExternalFiles) {
    FAIL() << "TODO: cover PathNotValid, InternalError, and ModelParseError";
}

TEST(FireReaderTest, DISABLED_TensorKeepsMappingAliveAfterReaderDestruction) {
    FAIL() << "TODO: construct Tensor with mapped_buffer and absolute byte offset";
}
