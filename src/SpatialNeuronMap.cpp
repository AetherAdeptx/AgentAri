#include "agentari/SpatialNeuronMap.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace agentari::text::hierarchical {
namespace {

constexpr double golden_angle = 2.39996322972865332;
constexpr std::size_t geometric_probe_limit = 4096U;
constexpr std::size_t spacing_candidate_count = 32U;
constexpr std::int64_t nearest_distance_search_radius = 8;

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
    ideal_isometric_spacing_ = 0.0;
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
    // Logarithmic Fibonacci growth keeps the spiral useful for large layer
    // counts without overflowing signed coordinate arithmetic.
    const double distance = std::min(
        static_cast<double>(spatial_grid_width) * 0.47,
        4.0 + 4.0 * std::log1p(static_cast<double>(spacing_term)));
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
    if (neurons_.empty()) {
        return preferred;
    }

    // Evaluate a deterministic candidate set instead of accepting the first
    // free cell. The score is measured against every neuron already inserted,
    // including fixed and layer-biased entries, so later allocations adapt to
    // the actual occupied geometry.
    SpatialCoordinate best = preferred;
    std::int64_t best_score = -1;
    std::size_t best_linear_index = spatial_grid_cell_count;
    bool found = false;
    const std::size_t probe_limit = std::min<std::size_t>(
        geometric_probe_limit, std::min(spatial_grid_cell_count, spacing_candidate_count));
    for (std::size_t probe = 0U; probe <= probe_limit; ++probe) {
        SpatialCoordinate candidate = preferred;
        if (probe != 0U) {
            // Collisions rotate counter-clockwise by the golden angle while
            // the probe radius grows. The z motion alternates, making the
            // search genuinely 3D.
            const double angle = golden_angle * static_cast<double>(local_index + probe);
            const double radius = 1.0 + std::sqrt(static_cast<double>(probe));
            const std::int64_t z_offset = static_cast<std::int64_t>(
                std::ceil(std::sqrt(static_cast<double>(probe)) * 0.5));
            candidate = SpatialCoordinate{
                .x = bounce_axis(static_cast<std::int64_t>(preferred.x) +
                                 rounded(radius * std::cos(angle))),
                .y = bounce_axis(static_cast<std::int64_t>(preferred.y) +
                                 rounded(radius * std::sin(angle))),
                .z = bounce_axis(static_cast<std::int64_t>(preferred.z) +
                                 ((probe & 1U) == 0U ? z_offset : -z_offset)),
            };
        }
        if (occupied(candidate)) {
            continue;
        }
        const std::int64_t score = nearest_occupied_distance_squared(candidate);
        const std::size_t candidate_index = linear_index(candidate);
        if (!found || score > best_score ||
            (score == best_score && candidate_index < best_linear_index)) {
            best = candidate;
            best_score = score;
            best_linear_index = candidate_index;
            found = true;
        }
    }
    if (found) {
        return best;
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

std::int64_t SpatialNeuronMap::nearest_occupied_distance_squared(
    const SpatialCoordinate coordinate,
    const SpatialNeuronId ignored_neuron) const noexcept {
    if (neurons_.empty()) {
        return std::numeric_limits<std::int64_t>::max();
    }

    const auto inspect_cube = [&](const std::int64_t radius) noexcept {
        const std::uint16_t minimum_x = coordinate.x > radius
                                            ? static_cast<std::uint16_t>(coordinate.x - radius)
                                            : 0U;
        const std::uint16_t minimum_y = coordinate.y > radius
                                            ? static_cast<std::uint16_t>(coordinate.y - radius)
                                            : 0U;
        const std::uint16_t minimum_z = coordinate.z > radius
                                            ? static_cast<std::uint16_t>(coordinate.z - radius)
                                            : 0U;
        const std::uint16_t maximum_x = static_cast<std::uint16_t>(std::min<std::int64_t>(
            static_cast<std::int64_t>(spatial_grid_width) - 1,
            static_cast<std::int64_t>(coordinate.x) + radius));
        const std::uint16_t maximum_y = static_cast<std::uint16_t>(std::min<std::int64_t>(
            static_cast<std::int64_t>(spatial_grid_width) - 1,
            static_cast<std::int64_t>(coordinate.y) + radius));
        const std::uint16_t maximum_z = static_cast<std::uint16_t>(std::min<std::int64_t>(
            static_cast<std::int64_t>(spatial_grid_width) - 1,
            static_cast<std::int64_t>(coordinate.z) + radius));

        std::int64_t best = std::numeric_limits<std::int64_t>::max();
        for (std::uint16_t z = minimum_z; z <= maximum_z; ++z) {
            for (std::uint16_t y = minimum_y; y <= maximum_y; ++y) {
                for (std::uint16_t x = minimum_x; x <= maximum_x; ++x) {
                    const GridCell& cell = cells_[linear_index(SpatialCoordinate{x, y, z})];
                    for (SpatialNeuronId id = cell.first_neuron;
                         id != invalid_spatial_neuron_id;
                         id = next_in_cell_[static_cast<std::size_t>(id)]) {
                        if (id != ignored_neuron) {
                            best = std::min(best, distance_squared(
                                                       coordinate,
                                                       neurons_[static_cast<std::size_t>(id)]
                                                           .coordinate));
                        }
                    }
                }
            }
        }
        return best;
    };

    for (std::int64_t radius = 0; radius <= nearest_distance_search_radius; ++radius) {
        const std::int64_t nearest = inspect_cube(radius);
        if (nearest != std::numeric_limits<std::int64_t>::max()) {
            return nearest;
        }
    }

    // Sparse maps can have neighbors farther away than the local search
    // radius. The fallback preserves exactness for metrics and deterministic
    // spacing decisions without making the common dense case expensive.
    std::int64_t best = std::numeric_limits<std::int64_t>::max();
    for (const SpatialNeuron& neuron : neurons_) {
        if (neuron.id != ignored_neuron) {
            best = std::min(best, distance_squared(coordinate, neuron.coordinate));
        }
    }
    return best;
}

void SpatialNeuronMap::insert(const SpatialNeuron neuron) {
    const std::size_t cell_index = linear_index(neuron.coordinate);
    GridCell& cell = cells_[cell_index];
    if (cell.count != 0U) {
        throw std::logic_error("attempted to overlap neurons in one spatial cell");
    }
    ++occupied_cells_;
    cell.first_neuron = neuron.id;
    cell.count = 1U;
    next_in_cell_.push_back(invalid_spatial_neuron_id);
    neurons_.push_back(neuron);
}

void SpatialNeuronMap::allocate(const std::span<const std::size_t> layer_neuron_counts) {
    allocate(layer_neuron_counts, std::span<const SpatialNeuronAssignment>{});
}

void SpatialNeuronMap::allocate(
    const std::span<const std::size_t> layer_neuron_counts,
    const std::span<const SpatialNeuronAssignment> assignments) {
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
    ideal_isometric_spacing_ = std::cbrt(
        static_cast<double>(spatial_grid_cell_count) / static_cast<double>(total));

    std::vector<std::size_t> assigned_by_layer(layer_neuron_counts.size(), 0U);
    if (assignments.size() > total) {
        throw std::invalid_argument("typed spatial assignments exceed the neuron count");
    }
    for (const SpatialNeuronAssignment& assignment : assignments) {
        if (assignment.type_number == 0U || assignment.layer_id >= assigned_by_layer.size()) {
            throw std::invalid_argument("typed spatial assignment has an invalid type or layer");
        }
        ++assigned_by_layer[assignment.layer_id];
    }
    for (std::size_t layer = 0U; layer < assigned_by_layer.size(); ++layer) {
        if (assigned_by_layer[layer] > layer_neuron_counts[layer]) {
            throw std::invalid_argument("typed spatial assignments exceed a layer capacity");
        }
    }

    neurons_.reserve(total);
    next_in_cell_.reserve(total);
    layers_.reserve(layer_neuron_counts.size());
    for (std::size_t layer = 0U; layer < layer_neuron_counts.size(); ++layer) {
        layers_.push_back(make_region(
            static_cast<std::uint32_t>(layer), layer_neuron_counts[layer], layer,
            layer_neuron_counts.size()));
    }

    std::vector<std::size_t> next_local_index(layer_neuron_counts.size(), 0U);
    const auto insert_layer_neuron = [&](const std::uint32_t layer_id,
                                         const std::uint32_t type_number) {
        const std::size_t layer = static_cast<std::size_t>(layer_id);
        const std::size_t local_index = next_local_index[layer]++;
        const SpatialLayerRegion& region = layers_[layer];
        const SpatialCoordinate preferred = preferred_coordinate(region, local_index);
        const SpatialCoordinate coordinate = find_free_coordinate(preferred, local_index);
        // Reserve the free cell before constructing/inserting the neuron
        // record. This ordering makes overlap a hard invariant.
        insert(SpatialNeuron{
            .id = static_cast<SpatialNeuronId>(neurons_.size()),
            .type_number = type_number,
            .layer_id = region.layer_id,
            .layer_local_index = static_cast<SpatialNeuronId>(local_index),
            .coordinate = coordinate,
            .position = PackedSpatialPosition::pack(coordinate),
        });
    };

    // Preserve the allocator's phase ordering in physical insertion order:
    // all explicitly typed fixed/percentage neurons are placed before any
    // unassigned remainder. This lets later percentage neurons see every
    // earlier fixed anchor during spacing decisions.
    for (const SpatialNeuronAssignment& assignment : assignments) {
        insert_layer_neuron(assignment.layer_id, assignment.type_number);
    }
    for (std::size_t layer = 0U; layer < layer_neuron_counts.size(); ++layer) {
        while (next_local_index[layer] < layers_[layer].neuron_count) {
            insert_layer_neuron(static_cast<std::uint32_t>(layer), 0U);
        }
    }
}

SpatialDistributionMetrics SpatialNeuronMap::distribution_metrics() const {
    SpatialDistributionMetrics metrics{
        .ideal_isometric_spacing = ideal_isometric_spacing_,
        .average_nearest_neighbor_distance = 0.0,
    };
    if (neurons_.size() < 2U) {
        return metrics;
    }
    long double total_distance = 0.0L;
    for (const SpatialNeuron& neuron : neurons_) {
        const std::int64_t nearest = nearest_occupied_distance_squared(
            neuron.coordinate, neuron.id);
        if (nearest != std::numeric_limits<std::int64_t>::max()) {
            total_distance += std::sqrt(static_cast<long double>(nearest));
        }
    }
    metrics.average_nearest_neighbor_distance = static_cast<double>(
        total_distance / static_cast<long double>(neurons_.size()));
    return metrics;
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
