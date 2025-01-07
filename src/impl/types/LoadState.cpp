#include "milvus/types/LoadState.h"

namespace milvus {

LoadState::LoadState() : code_(LoadStateCode::NotExist) {
    UpdateStateDesc();
}

LoadState::LoadState(LoadStateCode code) : code_(code) {
    UpdateStateDesc();
}

LoadStateCode LoadState::GetCode() const {
    return code_;
}

const std::string& LoadState::GetDesc() const {
    return state_desc_;
}

void LoadState::SetCode(LoadStateCode code) {
    code_ = code;
    UpdateStateDesc();
}

void LoadState::UpdateStateDesc() {
    switch (code_) {
        case LoadStateCode::NotExist:
            state_desc_ = "NotExist";
            break;
        case LoadStateCode::NotLoad:
            state_desc_ = "NotLoad";
            break;
        case LoadStateCode::Loading:
            state_desc_ = "Loading";
            break;
        case LoadStateCode::Loaded:
            state_desc_ = "Loaded";
            break;
        default:
            state_desc_ = "Unknown";
    }
}

}  // namespace milvus
