#pragma once
#include "milvus/v2/response/BaseResponse.h"
#include <vector>
#include <string>

namespace milvus {
class ListCollectionsResp : public BaseResponse {
public:
    ListCollectionsResp() = default;
    ListCollectionsResp(const Status& status, const std::vector<std::string>& collections)
        : BaseResponse(status), collections_(collections) {}
    const std::vector<std::string>& collections() const { return collections_; }
    void set_collections(const std::vector<std::string>& collections) { collections_ = collections; }
private:
    std::vector<std::string> collections_;
};
} // namespace milvus
