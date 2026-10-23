// Expose LuisaRender's existing orthographic camera without modifying Genesis.
#include <apps/app_base.h>
#include <apps/py_scene.h>

class PyWorldSolverOrtho final : public PyCamera {
public:
    PyWorldSolverOrtho(std::string_view name, PyTransform *pose, PyFilm *film,
                       PyFilter *filter, uint spp, float zoom) noexcept
        : PyCamera{name, "ortho", pose, film, filter, spp} {
        _node->add_property("zoom", zoom);
    }
};

PYBIND11_MODULE(WorldSolverLuisaExtras, m) {
    py::module_::import("LuisaRenderPy");
    py::class_<PyWorldSolverOrtho, PyCamera>(m, "OrthographicCamera")
        .def(py::init<std::string_view, PyTransform *, PyFilm *, PyFilter *, uint, float>(),
             py::arg("name"), py::arg("pose"), py::arg("film"), py::arg("filter"),
             py::arg("spp"), py::arg("zoom"));
}
