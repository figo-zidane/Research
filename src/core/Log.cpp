#include "core/Log.h"

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace rr::core
{
namespace
{
std::shared_ptr<spdlog::logger> g_logger;
}

void initialize_logging()
{
    if (g_logger)
    {
        return;
    }

    g_logger = spdlog::stdout_color_mt("research-renderer");
    g_logger->set_pattern("[%T] [%^%l%$] %v");
    g_logger->set_level(spdlog::level::info);
    spdlog::set_default_logger(g_logger);
}

std::shared_ptr<spdlog::logger> log()
{
    if (!g_logger)
    {
        initialize_logging();
    }

    return g_logger;
}
}
