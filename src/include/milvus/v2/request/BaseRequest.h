#pragma once
#include <cstdint>

namespace milvus {
class BaseRequest {
public:
    explicit BaseRequest(uint64_t timeout = 10000) : timeout_(timeout) {}
    virtual ~BaseRequest() = default;
    uint64_t timeout() const { return timeout_; }
    void set_timeout(uint64_t timeout) { timeout_ = timeout; }
protected:
    uint64_t timeout_;
};
} // namespace milvus
 