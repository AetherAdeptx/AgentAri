#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace agentari::text::hierarchical {

// The first spatial experiment deliberately uses a fixed power-of-two cube.
// It is large enough to show layer geometry while keeping coordinate math
// cheap and predictable.
inline constexpr std::uint16_t spatial_grid_width = 128U;
inline constexpr std::size_t spatial_grid_cell_count =
    static_cast<std::size_t>(spatial_grid_width) * spatial_grid_width * spatial_grid_width;
inline constexpr std::uint32_t spatial_any_layer = std::numeric_limits<std::uint32_t>::max();
using SpatialNeuronId = std::uint64_t;
inline constexpr SpatialNeuronId invalid_spatial_neuron_id =
    std::numeric_limits<SpatialNeuronId>::max();

struct SpatialCoordinate {
    std::uint16_t x{0U};
    std::uint16_t y{0U};
    std::uint16_t z{0U};

    [[nodiscard]] constexpr bool operator==(const SpatialCoordinate&) const noexcept = default;
};

// A packed position reserves 16 bits for each coordinate and 16 bits for
// future color/cluster data. It is an address/label handle, not the neuron
// weights or activation payload.
struct PackedSpatialPosition {
    std::uint64_t value{0U};

    [[nodiscard]] static constexpr PackedSpatialPosition pack(
        const SpatialCoordinate coordinate, const std::uint16_t color = 0U) noexcept {
        return PackedSpatialPosition{
            static_cast<std::uint64_t>(coordinate.x) |
            (static_cast<std::uint64_t>(coordinate.y) << 16U) |
            (static_cast<std::uint64_t>(coordinate.z) << 32U) |
            (static_cast<std::uint64_t>(color) << 48U),
        };
    }

    [[nodiscard]] constexpr SpatialCoordinate coordinate() const noexcept {
        return SpatialCoordinate{
            .x = static_cast<std::uint16_t>(value & 0xffffU),
            .y = static_cast<std::uint16_t>((value >> 16U) & 0xffffU),
            .z = static_cast<std::uint16_t>((value >> 32U) & 0xffffU),
        };
    }

    [[nodiscard]] constexpr std::uint16_t color() const noexcept {
        return static_cast<std::uint16_t>((value >> 48U) & 0xffffU);
    }
};

struct SpatialLayerRegion {
    std::uint32_t layer_id{0U};
    std::size_t neuron_count{0U};
    SpatialCoordinate anchor{};
    PackedSpatialPosition position{};
    std::uint16_t width{0U};
    std::uint16_t height{0U};
    std::uint16_t depth{0U};
};

struct SpatialNeuron {
    SpatialNeuronId id{0U};
    std::uint32_t layer_id{0U};
    SpatialNeuronId layer_local_index{0U};
    SpatialCoordinate coordinate{};
    PackedSpatialPosition position{};
};

struct SpatialSearchQuery {
    SpatialCoordinate center{};
    std::uint16_t radius{0U};
    std::uint32_t layer_id{spatial_any_layer};
    std::size_t max_results{std::numeric_limits<std::size_t>::max()};
};

class SpatialNeuronMap {
public:
    using SearchVisitor = std::function<bool(const SpatialNeuron&)>;

    SpatialNeuronMap();

    // Rebuilds the map from layer sizes. The span index is the layer ID.
    // Every neuron receives a coordinate, while a cell remains capable of
    // holding multiple neurons for future compressed or over-subscribed maps.
    void allocate(std::span<const std::size_t> layer_neuron_counts);
    void clear();

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t occupied_cell_count() const noexcept;
    [[nodiscard]] std::size_t layer_count() const noexcept;
    [[nodiscard]] const std::vector<SpatialNeuron>& neurons() const noexcept;
    [[nodiscard]] const std::vector<SpatialLayerRegion>& layers() const noexcept;
    [[nodiscard]] const SpatialNeuron* find(SpatialNeuronId id) const noexcept;
    [[nodiscard]] const SpatialNeuron* find(SpatialCoordinate coordinate) const noexcept;

    // Smart search materializes nearest-first results. Exact and small-radius
    // queries use the dense occupancy grid, so they do not scan all neurons.
    [[nodiscard]] std::vector<SpatialNeuron> smart_search(
        const SpatialSearchQuery& query) const;

    // Lazy search invokes the visitor as cells are visited and allocates no
    // result vector. Returning false stops the search early.
    [[nodiscard]] std::size_t lazy_search(const SpatialSearchQuery& query,
                                           const SearchVisitor& visitor) const;

private:
    struct GridCell {
        SpatialNeuronId first_neuron{invalid_spatial_neuron_id};
        SpatialNeuronId count{0U};
    };

    std::vector<GridCell> cells_;
    std::vector<SpatialNeuron> neurons_;
    std::vector<SpatialNeuronId> next_in_cell_;
    std::vector<SpatialLayerRegion> layers_;
    std::size_t occupied_cells_{0U};

    [[nodiscard]] static std::size_t linear_index(SpatialCoordinate coordinate) noexcept;
    [[nodiscard]] static std::uint16_t bounce_axis(std::int64_t value) noexcept;
    [[nodiscard]] static std::uint64_t fibonacci(std::size_t index) noexcept;
    [[nodiscard]] static SpatialCoordinate layer_anchor(std::size_t layer_index,
                                                         std::size_t layer_count) noexcept;
    [[nodiscard]] static SpatialLayerRegion make_region(std::uint32_t layer_id,
                                                         std::size_t neuron_count,
                                                         std::size_t layer_index,
                                                         std::size_t layer_count);
    [[nodiscard]] SpatialCoordinate preferred_coordinate(const SpatialLayerRegion& region,
                                                         std::size_t local_index) const noexcept;
    [[nodiscard]] SpatialCoordinate find_free_coordinate(SpatialCoordinate preferred,
                                                          std::size_t local_index) const;
    [[nodiscard]] bool occupied(SpatialCoordinate coordinate) const noexcept;
    void insert(SpatialNeuron neuron);
    [[nodiscard]] std::size_t visit_query(const SpatialSearchQuery& query,
                                          const SearchVisitor& visitor) const;
};

}  // namespace agentari::text::hierarchical
