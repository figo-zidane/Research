#pragma once

#include <memory>

#include <spdlog/logger.h>

namespace rr::core
{
void initialize_logging();
std::shared_ptr<spdlog::logger> log();
}
