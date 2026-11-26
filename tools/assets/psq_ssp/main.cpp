// SPDX-License-Identifier: GPL-3.0-or-later
// Offline adapter for Liu et al.'s Surface Multigrid via Intrinsic Prolongation.
#include <SSP_decimate.h>
#include <query_fine_to_coarse.h>
#include <igl/read_triangle_mesh.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

using namespace Eigen;

struct Queries
{
    MatrixXd weights;
    MatrixXi vertices;
    VectorXi faces;
};

Queries vertex_queries(int count, const MatrixXi& faces)
{
    Queries result{MatrixXd::Zero(count, 3), MatrixXi(count, 3), VectorXi::Constant(count, -1)};
    for(int face = 0; face < faces.rows(); ++face)
        for(int corner = 0; corner < 3; ++corner)
        {
            const int vertex = faces(face, corner);
            if(result.faces(vertex) < 0)
            {
                result.faces(vertex) = face;
                result.vertices.row(vertex) = faces.row(face);
                result.weights(vertex, corner) = 1;
            }
        }
    if(result.faces.minCoeff() < 0)
        throw std::runtime_error("Mesh contains isolated vertices");
    return result;
}

template <class Matrix>
void write_matrix(const std::filesystem::path& path, const Matrix& matrix)
{
    std::ofstream out(path);
    out << std::setprecision(17) << matrix << '\n';
    if(!out)
        throw std::runtime_error("Could not write " + path.string());
}

int main(int argc, char** argv)
try
{
    if(argc < 5)
        throw std::runtime_error("Usage: psq_ssp fine.obj directory level_count coarse_face_counts...");
    const std::filesystem::path directory = argv[2];
    const int levels = std::stoi(argv[3]);
    if(levels < 2 || argc != levels + 3)
        throw std::runtime_error("Expected one target count per reduction, finest to coarsest");
    std::filesystem::create_directories(directory);

    MatrixXd positions;
    MatrixXi faces;
    if(!igl::read_triangle_mesh(argv[1], positions, faces) || faces.cols() != 3
       || positions.rows() == 0 || !positions.allFinite())
        throw std::runtime_error("Expected a finite triangle mesh");
    auto prefix = [&](int level, std::string_view suffix)
    {
        return directory / ("L" + std::to_string(level) + std::string(suffix));
    };
    write_matrix(prefix(levels - 1, "_vertices.txt"), positions);
    write_matrix(prefix(levels - 1, "_triangles.txt"), faces);
    Queries finest = vertex_queries(positions.rows(), faces);
    for(int level = levels - 2; level >= 0; --level)
    {
        const int target = std::stoi(argv[levels + 2 - level]);
        if(target < 4 || target >= faces.rows())
            throw std::runtime_error("Face targets must be strictly decreasing");
        MatrixXd coarse_positions;
        MatrixXi coarse_faces;
        VectorXi birth_faces, birth_vertices, face_index;
        std::vector<single_collapse_data> collapses;
        std::vector<std::vector<int>> face_collapses;
        SSP_decimate(positions, faces, target, 0, coarse_positions, coarse_faces,
                     birth_faces, birth_vertices, collapses, face_collapses, face_index);
        if(coarse_faces.rows() > target || coarse_faces.rows() == 0)
            throw std::runtime_error("Decimation did not reach target face count");
        auto trace = [&](Queries& points)
        {
            query_fine_to_coarse(collapses, birth_vertices, face_collapses, face_index,
                                 points.weights, points.vertices, points.faces);
            if(points.faces.minCoeff() < 0 || points.faces.maxCoeff() >= coarse_faces.rows()
               || points.vertices.minCoeff() < 0 || points.vertices.maxCoeff() >= coarse_positions.rows()
               || !points.weights.allFinite())
                throw std::runtime_error("Invalid SSP correspondence");
        };
        Queries adjacent = vertex_queries(positions.rows(), faces);
        trace(adjacent);
        trace(finest);
        write_matrix(prefix(level, "_vertices.txt"), coarse_positions);
        write_matrix(prefix(level, "_triangles.txt"), coarse_faces);
        write_matrix(prefix(level, "_adjacent_vertices.txt"), adjacent.vertices);
        write_matrix(prefix(level, "_adjacent_weights.txt"), adjacent.weights);
        write_matrix(prefix(level, "_adjacent_faces.txt"), adjacent.faces);
        write_matrix(prefix(level, "_direct_faces.txt"), finest.faces);
        positions = std::move(coarse_positions);
        faces = std::move(coarse_faces);
    }
    return 0;
}
catch(const std::exception& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
