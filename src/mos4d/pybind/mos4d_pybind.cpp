// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <Eigen/Core>
#include <memory>
#include <vector>

#include "Deskew.hpp"
#include "stl_vector_eigen.h"

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3d>);

namespace mos4d {
PYBIND11_MODULE(mos4d_pybind, m) {
    auto vector3dvector = pybind_eigen_vector_of_vector<Eigen::Vector3d>(
        m, "_Vector3dVector", "std::vector<Eigen::Vector3d>",
        py::py_array_to_vectors_double<Eigen::Vector3d>);

    m.def(
        "_deskew",
        [](const std::vector<Eigen::Vector3d> &points, const std::vector<double> &timestamps,
           const Eigen::Matrix4d &relative_motion) {
            Sophus::SE3d motion(relative_motion);
            return Deskew(points, timestamps, motion);
        },
        "frame"_a, "timestamps"_a, "relative_motion"_a);
}
}  // namespace mos4d
