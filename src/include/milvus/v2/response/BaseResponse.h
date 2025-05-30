#pragma once
#include <string>
#include "milvus/Status.h"

namespace milvus {

class BaseResponse {
public:
    explicit BaseResponse(const Status& status = Status()) : status_(status) {}
    virtual ~BaseResponse() = default;
    const Status& GetStatus() const { return status_; }
    void SetStatus(const Status& status) { status_ = status; }
protected:
    Status status_;
};
} // namespace milvus
