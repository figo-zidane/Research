#include "app/Application.h"
#include "core/Log.h"

#include <cstdlib>
#include <exception>

int main()
{
    rr::core::initialize_logging();

    try
    {
        rr::app::Application app;
        app.run();
    }
    catch (const std::exception& exception)
    {
        rr::core::log()->critical("Unhandled exception: {}", exception.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
