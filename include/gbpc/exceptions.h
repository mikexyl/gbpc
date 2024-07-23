#ifndef GBPC_EXCEPTIONS_H_
#define GBPC_EXCEPTIONS_H_

#include <exception>
#include <string>

namespace gbpc {
class NodeNotFoundException : public std::exception {
public:
  NodeNotFoundException(size_t key) noexcept : key_(key) {
    msg_ = "Node with key " + std::to_string(key_) + " not found";
  }

  virtual const char *what() const noexcept override { return msg_.c_str(); }

  size_t key() const { return key_; }

  size_t key_;
  std::string msg_;
};

class NodeAlreadyExistsException : public std::exception {
public:
  NodeAlreadyExistsException(size_t key) noexcept : key_(key) {
    msg_ = "Node with key " + std::to_string(key_) + " not found";
  }

  virtual const char *what() const noexcept override { return msg_.c_str(); }

  size_t key() const { return key_; }

  size_t key_;
  std::string msg_;
};

} // namespace gbpc

#endif // GBPC_EXCEPTIONS_H_
