/**
 * Guards the interface layer's contract: the layer's own files (directly in
 * spacetime/interface/) create no threads and never sleep, block or wait, so each
 * interface -- one subfolder each, e.g. spacetime/interface/standalone/ -- decides how
 * threads are handled. Data shared across threads may still be guarded by a std::mutex.
 */
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace Spacetime::Interface
{
    TEST(InterfaceLayer, HasNoThreadHandling)
    {
        const std::vector<std::string> forbidden = {"<thread>", "<future>", "condition_variable", "stop_token",
                                                    "jthread", "std::thread", "sleep_for", "sleep_until", "std::async"};
        const std::filesystem::path dir = std::filesystem::path(PROJECT_SOURCE_DIR) / "spacetime" / "interface";
        ASSERT_TRUE(std::filesystem::is_directory(dir)) << dir;

        int files = 0;
        for (const auto &entry : std::filesystem::directory_iterator(dir))
        {
            if (!entry.is_regular_file()) // the interfaces' own subfolders handle their threads
                continue;
            files++;
            std::ifstream in(entry.path());
            std::stringstream contents;
            contents << in.rdbuf();
            for (const auto &token : forbidden)
                EXPECT_EQ(contents.str().find(token), std::string::npos)
                    << entry.path() << " uses " << token << "; thread handling belongs in an interface (e.g. spacetime/interface/standalone/)";
        }
        EXPECT_GT(files, 0);
    }
} // namespace Spacetime::Interface
