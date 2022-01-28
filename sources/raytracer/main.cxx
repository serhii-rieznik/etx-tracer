#include <etx/app/app.hxx>

namespace etx {

struct RaytracerApplication : public ApplicationImpl {};

}  // namespace etx

int main(int argc, char* argv[]) {
  etx::RaytracerApplication app;
  return etx::Application(app).run(argc, argv);
}
