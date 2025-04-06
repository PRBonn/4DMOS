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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "Deskew.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <functional>
#include <vector>

namespace mos4d {

std::vector<Eigen::Vector3d> Deskew(const std::vector<Eigen::Vector3d> &frame,
                                    const std::vector<double> &timestamps,
                                    const Sophus::SE3d &relative_motion) {
    const std::vector<Eigen::Vector3d> &deskewed_frame = [&]() {
        if (timestamps.empty()) {
            return frame;
        } else {
            const auto &[min, max] = std::minmax_element(timestamps.cbegin(), timestamps.cend());
            const double min_time = *min;
            const double max_time = *max;
            const auto normalize = [&](const double t) {
                return (t - min_time) / (max_time - min_time);
            };
            const auto &omega = relative_motion.log();
            std::vector<Eigen::Vector3d> deskewed_frame(frame.size());
            tbb::parallel_for(
                // Index Range
                tbb::blocked_range<size_t>{0, deskewed_frame.size()},
                // Parallel Compute
                [&](const tbb::blocked_range<size_t> &r) {
                    for (size_t idx = r.begin(); idx < r.end(); ++idx) {
                        const auto &point = frame.at(idx);
                        const auto &stamp = normalize(timestamps.at(idx));
                        const auto pose = Sophus::SE3d::exp((stamp - 1.0) * omega);
                        deskewed_frame.at(idx) = pose * point;
                    };
                });
            return deskewed_frame;
        }
    }();
    return deskewed_frame;
}

}  // namespace mos4d
