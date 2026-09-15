#include "agentari/SpatialNeuronMap.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace agentari::text::hierarchical {
namespace {

constexpr double pi = 3.14159265358979323846;
constexpr double golden_angle = 2.39996322972865332;
constexpr std::size_t geometric_probe_limit = 4096U;

std::size_t ceil_size(const double value) {
    return static_cast<std::size_t>(std::ceil(value));
}

std::uint16_t dimension(const std::size_t value) {
    return static_cast<std::uint16_t>(std::clamp<std::size_t>(value, 1U,
                                                               spatial_grid_width));
}

std::int64_t rounded(const double value) {
    return static_cast<std::int64_t>(std::llround(value));
}

std::int64_t distance_squared(const SpatialCoordinate left, const SpatialCoordinate right) {
    const std::int64_t dx = static_cast<std::int64_t>(left.x) -
                            static_cast<std::int64_t>(right.x);
    const std::int64_t dy = static_cast<std::int64_t>(left.y) -
                            static_cast<std::int64_t>(right.y);
    const std::int64_t dz = static_cast<std::int64_t>(left.z) -
                            static_cast<std::int64_t>(right.z);
    return dx * dx + dy * dy + dz * dz;
}

}  // namespace

SpatialNeuronMap::SpatialNeuronMap() : cells_(spatial_grid_cell_count) {}

void SpatialNeuronMap::clear() {
    std::fill(cells_.begin(), cells_.end(), GridCell{});
    neurons_.clear();
    next_in_cell_.clear();
    layers_.clear();
    occupied_cells_ = 0U;
}

std::size_t SpatialNeuronMap::size() const noexcept {
    return neurons_.size();
}

std::size_t SpatialNeuronMap::occupied_cell_count() const noexcept {
    return occupied_cells_;
}

std::size_t SpatialNeuronMap::layer_count() const noexcept {
    return layers_.size();
}

const std::vector<SpatialNeuron>& SpatialNeuronMap::neurons() const noexcept {
    return neurons_;
}

const std::vector<SpatialLayerRegion>& SpatialNeuronMap::layers() const noexcept {
    return layers_;
}

std::size_t SpatialNeuronMap::linear_index(const SpatialCoordinate coordinate) noexcept {
    return static_cast<std::size_t>(coordinate.x) +
           static_cast<std::size_t>(spatial_grid_width) *
               (static_cast<std::size_t>(coordinate.y) +
                static_cast<std::size_t>(spatial_grid_width) * coordinate.z);
}

std::uint16_t SpatialNeuronMap::bounce_axis(const std::int64_t value) noexcept {
    constexpr std::int64_t last = static_cast<std::int64_t>(spatial_grid_width) - 1;
    constexpr std::int64_t period = last * 2;
    std::int64_t wrapped = value % period;
    if (wrapped < 0) {
        wrapped += period;
    }
    if (wrapped > last) {
        wrapped = period - wrapped;
    }
    return static_cast<std::uint16_t>(wrapped);
}

std::uint64_t SpatialNeuronMap::fibonacci(const std::size_t index) noexcept {
    if (index == 0U) {
        return 0U;
    }
    std::uint64_t previous = 0U;
    std::uint64_t current = 1U;
    for (std::size_t step = 1U; step < index; ++step) {
        if (current > std::numeric_limits<std::uint64_t>::max() - previous) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        const std::uint64_t next = previous + current;
        previous = current;
        current = next;
    }
    return current;
}

SpatialCoordinate SpatialNeuronMap::layer_anchor(const std::size_t layer_index,
                                                 const std::size_t layer_count) noexcept {
    const double normalized_index = static_cast<double>(layer_index) + 0.5;
    const double normalized_count = std::max(1.0, static_cast<double>(layer_count));
    // Fibonacci-sphere elevation plus golden-angle rotation gives a stable
    // three-dimensional analogue of a phyllotaxis spiral.
    const double z = 1.0 - 2.0 * normalized_index / normalized_count;
    const double radial = std::sqrt(std::max(0.0, 1.0 - z * z));
    const double theta = golden_angle * static_cast<double>(layer_index + 1U);
    const std::uint64_t spacing_term = fibonacci(layer_index + 4U);
    const double distance = 4.0 + 2.0 * static_cast<double>(spacing_term);
    const double center = (static_cast<double>(spatial_grid_width) - 1.0) * 0.5;
    return SpatialCoordinate{
        .x = bounce_axis(rounded(center + distance * radial * std::cos(theta))),
        .y = bounce_axis(rounded(center + distance * radial * std::sin(theta))),
        .z = bounce_axis(rounded(center + distance * z)),
    };
}

SpatialLayerRegion SpatialNeuronMap::make_region(const std::uint32_t layer_id,
                                                 const std::size_t neuron_count,
                                                 const std::size_t layer_index,
                                                 const std::size_t layer_count) {
    const std::size_t count_for_shape = std::max<std::size_t>(1U, neuron_count);
    const std::size_t width = std::min<std::size_t>(
        spatial_grid_width, ceil_size(std::cbrt(static_cast<double>(count_for_shape))));
    const std::size_t rows =
        (count_for_shape + width - 1U) / width;
    const std::size_t height = std::min<std::size_t>(
        spatial_grid_width, ceil_size(std::sqrt(static_cast<double>(rows))));
    const std::size_t depth = std::min<std::size_t>(
        spatial_grid_width, (count_for_shape + width * height - 1U) / (width * height));
    const SpatialCoordinate anchor = layer_anchor(layer_index, layer_count);
    return SpatialLayerRegion{
        .layer_id = layer_id,
        .neuron_count = neuron_count,
        .anchor = anchor,
        .position = PackedSpatialPosition::pack(anchor),
        .width = dimension(width),
        .height = dimension(height),
        .depth = dimension(depth),
    };
}

SpatialCoordinate SpatialNeuronMap::preferred_coordinate(const SpatialLayerRegion& region,
                                                          const std::size_t local_index) const
    noexcept {
    const std::size_t width = std::max<std::size_t>(1U, region.width);
    const std::size_t height = std::max<std::size_t>(1U, region.height);
    const std::size_t x_index = local_index % width;
    const std::size_t y_index = (local_index / width) % height;
    const std::size_t z_index = (local_index / (width * height)) %
                                std::max<std::size_t>(1U, region.depth);
    const std::int64_t x_offset = static_cast<std::int64_t>(x_index) -
                                  static_cast<std::int64_t>((width - 1U) / 2U);
    const std::int64_t y_offset = static_cast<std::int64_t>(y_index) -
                                  static_cast<std::int64_t>((height - 1U) / 2U);
    const std::int64_t z_offset = static_cast<std::int64_t>(z_index) -
                                  static_cast<std::int64_t>((region.depth - 1U) / 2U);
    return SpatialCoordinate{
        .x = bounce_axis(static_cast<std::int64_t>(region.anchor.x) + x_offset),
        .y = bounce_axis(static_cast<std::int64_t>(region.anchor.y) + y_offset),
        .z = bounce_axis(static_cast<std::int64_t>(region.anchor.z) + z_offset),
    };
}

SpatialCoordinate SpatialNeuronMap::find_free_coordinate(const SpatialCoordinate preferred,
                                                          const std::size_t local_index) const {
    if (!occupied(preferred)) {
        return preferred;
    }

    // Collisions rotate counter-clockwise by the golden angle while the probe
    // radius grows. The z motion alternates, making the search genuinely 3D.
    const std::size_t probe_limit = std::min<std::size_t>(geometric_probe_limit,
                                                           spatial_grid_cell_count);
    for (std::size_t probe = 1U; probe <= probe_limit; ++probe) {
        const double angle = golden_angle * static_cast<double>(local_index + probe);
        const double radius = 1.0 + std::sqrt(static_cast<double>(probe));
        const std::int64_t z_offset = static_cast<std::int64_t>(
            std::ceil(std::sqrt(static_cast<double>(probe)) * 0.5));
        const SpatialCoordinate candidate{
            .x = bounce_axis(static_cast<std::int64_t>(preferred.x) +
                             rounded(radius * std::cos(angle))),
            .y = bounce_axis(static_cast<std::int64_t>(preferred.y) +
                             rounded(radius * std::sin(angle))),
            .z = bounce_axis(static_cast<std::int64_t>(preferred.z) +
                             ((probe & 1U) == 0U ? z_offset : -z_offset)),
        };
        if (!occupied(candidate)) {
            return candidate;
        }
    }

    // The geometric probe is the readable path. This deterministic fallback
    // guarantees termination even when a dense cluster defeats the spiral.
    const std::size_t start = linear_index(preferred);
    for (std::size_t offset = 1U; offset < spatial_grid_cell_count; ++offset) {
        const std::size_t index = (start + offset) % spatial_grid_cell_count;
        const SpatialCoordinate candidate{
            .x = static_cast<std::uint16_t>(index % spatial_grid_width),
            .y = static_cast<std::uint16_t>((index / spatial_grid_width) % spatial_grid_width),
            .z = static_cast<std::uint16_t>(index / (spatial_grid_width * spatial_grid_width)),
        };
        if (!occupied(candidate)) {
            return candidate;
        }
    }
    throw std::runtime_error("the 128 cubed spatial neuron grid is full");
}

bool SpatialNeuronMap::occupied(const SpatialCoordinate coordinate) const noexcept {
    return cells_[linear_index(coordinate)].count != 0U;
}

void SpatialNeuronMap::insert(const SpatialNeuron neuron) {
    const std::size_t cell_index = linear_index(neuron.coordinate);
    GridCell& cell = cells_[cell_index];
    if (cell.count == 0U) {
        ++occupied_cells_;
    }
    next_in_cell_.push_back(cell.first_neuron);
    cell.first_neuron = neuron.id;
    ++cell.count;
    neurons_.push_back(neuron);
}

void SpatialNeuronMap::allocate(const std::span<const std::size_t> layer_neuron_counts) {
    if (layer_neuron_counts.empty()) {
        clear();
        return;
    }

    std::size_t total = 0U;
    for (const std::size_t count : layer_neuron_counts) {
        if (count > std::numeric_limits<std::size_t>::max() - total) {
            throw std::invalid_argument("spatial neuron count exceeds host index capacity");
        }
        total += count;
    }

    clear();
    neurons_.reserve(total);
    next_in_cell_.reserve(total);
    layers_.reserve(layer_neuron_counts.size());
    for (std::size_t layer = 0U; layer < layer_neuron_counts.size(); ++layer) {
        const SpatialLayerRegion region = make_region(
            static_cast<std::uint32_t>(layer), layer_neuron_counts[layer], layer,
            layer_neuron_counts.size());
        layers_.push_back(region);
        for (std::size_t local_index = 0U; local_index < region.neuron_count; ++local_index) {
            const SpatialCoordinate preferred = preferred_coordinate(region, local_index);
            const SpatialCoordinate coordinate = find_free_coordinate(preferred, local_index);
            insert(SpatialNeuron{
                .id = static_cast<SpatialNeuronId>(neurons_.size()),
                .layer_id = region.layer_id,
                .layer_local_index = static_cast<SpatialNeuronId>(local_index),
                .coordinate = coordinate,
                .position = PackedSpatialPosition::pack(coordinate),
            });
        }
    }
}

const SpatialNeuron* SpatialNeuronMap::find(const SpatialNeuronId id) const noexcept {
    return id < neurons_.size() ? &neurons_[id] : nullptr;
}

const SpatialNeuron* SpatialNeuronMap::find(const SpatialCoordinate coordinate) const noexcept {
    const GridCell& cell = cells_[linear_index(coordinate)];
    return cell.first_neuron == invalid_spatial_neuron_id
               ? nullptr
               : &neurons_[static_cast<std::size_t>(cell.first_neuron)];
}

std::size_t SpatialNeuronMap::visit_query(const SpatialSearchQuery& query,
                                          const SearchVisitor& visitor) const {
    if (!visitor || query.max_results == 0U) {
        return 0U;
    }
    const std::int64_t radius = static_cast<std::int64_t>(query.radius);
    const std::int64_t radius_squared = radius * radius;
    const std::uint16_t minimum_x = query.center.x > query.radius
                                        ? static_cast<std::uint16_t>(query.center.x - query.radius)
                                        : 0U;
    const std::uint16_t minimum_y = query.center.y > query.radius
                                        ? static_cast<std::uint16_t>(query.center.y - query.radius)
                                        : 0U;
    const std::uint16_t minimum_z = query.center.z > query.radius
                                        ? static_cast<std::uint16_t>(query.center.z - query.radius)
                                        : 0U;
    const std::uint16_t maximum_x = static_cast<std::uint16_t>(std::min<std::size_t>(
        spatial_grid_width - 1U, static_cast<std::size_t>(query.center.x) + query.radius));
    const std::uint16_t maximum_y = static_cast<std::uint16_t>(std::min<std::size_t>(
        spatial_grid_width - 1U, static_cast<std::size_t>(query.center.y) + query.radius));
    const std::uint16_t maximum_z = static_cast<std::uint16_t>(std::min<std::size_t>(
        spatial_grid_width - 1U, static_cast<std::size_t>(query.center.z) + query.radius));

    std::size_t delivered = 0U;
    for (std::uint16_t z = minimum_z; z <= maximum_z; ++z) {
        for (std::uint16_t y = minimum_y; y <= maximum_y; ++y) {
            for (std::uint16_t x = minimum_x; x <= maximum_x; ++x) {
                const SpatialCoordinate coordinate{x, y, z};
                const GridCell& cell = cells_[linear_index(coordinate)];
                for (SpatialNeuronId id = cell.first_neuron; id != invalid_spatial_neuron_id;
                     id = next_in_cell_[static_cast<std::size_t>(id)]) {
                    const SpatialNeuron& neuron = neurons_[static_cast<std::size_t>(id)];
                    if ((query.layer_id == spatial_any_layer ||
                         neuron.layer_id == query.layer_id) &&
                        distance_squared(neuron.coordinate, query.center) <= radius_squared) {
                        ++delivered;
                        if (!visitor(neuron) || delivered >= query.max_results) {
                            return delivered;
                        }
                    }
                }
            }
        }
    }
    return delivered;
}

std::size_t SpatialNeuronMap::lazy_search(const SpatialSearchQuery& query,
                                           const SearchVisitor& visitor) const {
    return visit_query(query, visitor);
}

std::vector<SpatialNeuron> SpatialNeuronMap::smart_search(const SpatialSearchQuery& query) const {
    std::vector<SpatialNeuron> result;
    const SpatialSearchQuery complete_query{
        .center = query.center,
        .radius = query.radius,
        .layer_id = query.layer_id,
        .max_results = std::numeric_limits<std::size_t>::max(),
    };
    (void)visit_query(complete_query, [&result](const SpatialNeuron& neuron) {
        result.push_back(neuron);
        return true;
    });
    std::sort(result.begin(), result.end(), [&query](const SpatialNeuron& left,
                                                     const SpatialNeuron& right) {
        const std::int64_t left_distance = distance_squared(left.coordinate, query.center);
        const std::int64_t right_distance = distance_squared(right.coordinate, query.center);
        if (left_distance != right_distance) {
            return left_distance < right_distance;
        }
        return left.id < right.id;
    });
    if (result.size() > query.max_results) {
        result.resize(query.max_results);
    }
    return result;
}

}  // namespace agentari::text::hierarchical
