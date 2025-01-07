#pragma once

#include <string>

namespace milvus {

enum class LoadStateCode {
    NotExist = 0,
    NotLoad = 1,
    Loading = 2,
    Loaded = 3
};

class LoadState {
public:
    LoadState();
    explicit LoadState(LoadStateCode code);

    LoadStateCode GetCode() const;
    const std::string& GetDesc() const;
    void SetCode(LoadStateCode code);

private:
    LoadStateCode code_;
    std::string state_desc_;
    void UpdateStateDesc();
};

}  // namespace milvus
