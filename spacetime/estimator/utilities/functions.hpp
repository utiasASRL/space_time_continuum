#ifndef UTILITY_FUNCTIONS_H
#define UTILITY_FUNCTIONS_H

#include <fstream>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <json/json.h>
#include <json/value.h>
#include <gtsam/nonlinear/Symbol.h>

#include <spacetime/estimator/utilities/logger.hpp>

#define MDEG2RAD 0.0000174532925
#define TOLERANCE 1e-8

/// A rows x rows zero matrix.
inline Eigen::MatrixXd ZERO(int rows) { return Eigen::MatrixXd::Zero(rows, rows); }
/// A rows x rows identity matrix.
inline Eigen::MatrixXd EYE(int rows) { return Eigen::MatrixXd::Identity(rows, rows); }

/**
 * @brief Assert (debug builds only) that the 3x3 matrix C is orthonormal, to 1e-5.
 *        From Barfoot and Furgale, "Associating Uncertainty with Three-Dimensional Poses
 *        for use in Estimation Problems", DOI: 10.1109/TRO.2014.2298059.
 */
template <typename Derived>
void validateRotationMatrix(const Eigen::MatrixBase<Derived> &C)
{
    assert((C.cols() == 3 && C.rows() == 3) && "Input needs to be 3x3 matrix");

    const Eigen::Matrix<typename Derived::Scalar, 3, 3> CtC = C.transpose() * C;
    Eigen::Matrix<typename Derived::Scalar, 3, 3> E = CtC - Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity();
    E = E.array().abs();

    double error = E.maxCoeff();

    assert((error < 1e-5) && "C must be a valid rotation matrix");
}

/// Assert (debug builds only) that the 4x4 matrix T is in SE(3) (same source as validateRotationMatrix).
template <typename Derived>
void validateTransformationMatrix(const Eigen::MatrixBase<Derived> &T)
{
    assert((T.rows() == 4 && T.cols() == 4) && "Input must be a 4x4 matrix");

    const auto C = T.template block<3, 3>(0, 0);
    validateRotationMatrix(C);

    Eigen::Matrix<typename Derived::Scalar, 1, 4> bottom_T;
    bottom_T << 0, 0, 0, 1;

    Eigen::Matrix<typename Derived::Scalar, 1, 4> diff = bottom_T - T.bottomRows(1);
    diff = diff.array().abs();

    double error = diff.maxCoeff();
    assert((error < 1e-10) && "T must be a valid transformation matrix");
}

/// Declared but never defined; see the template overload below.
void validateDiagonalMatrix(const Eigen::MatrixXd &matrix);

/// Inverse of an SE(3) matrix, [C^T, -C^T r; 0, 1].
template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 4, 4> invertTransformation(const Eigen::MatrixBase<Derived> &mat)
{
    validateTransformationMatrix(mat); // Can remove (unsafe)

    Eigen::Matrix<typename Derived::Scalar, 4, 4> inv_T = Eigen::Matrix<typename Derived::Scalar, 4, 4>::Identity();
    Eigen::Matrix<typename Derived::Scalar, 3, 3> R_transpose = mat.template topLeftCorner<3, 3>().transpose();

    inv_T.template block<3, 3>(0, 0) = R_transpose;
    inv_T.template block<3, 1>(0, 3) = -R_transpose * mat.template block<3, 1>(0, 3);

    return inv_T;
}

/// Assert (debug builds only) that `matrix` is square and diagonal, to 1e-10.
template <typename Derived>
void validateDiagonalMatrix(const Eigen::MatrixBase<Derived> &matrix)
{
    assert((matrix.rows() == matrix.cols()) && "Input needs to be square matrix");

    bool is_diagonal = true;
    for (unsigned int i = 0; i < matrix.rows(); i++)
    {
        for (unsigned int j = 0; j < matrix.cols(); j++)
        {
            if (i != j && abs(matrix(i, j)) > 1e-10)
            {
                is_diagonal = false;
            }
        }
    }

    assert(is_diagonal && "Matrix can only have non-zero entries on diagonal");
}

/// Inverse of a diagonal matrix (element-wise reciprocal of the diagonal).
template <typename Derived>
Eigen::MatrixX<typename Derived::Scalar> invertDiagonal(const Eigen::MatrixBase<Derived> &matrix)
{
    validateDiagonalMatrix(matrix); // Can remove (unsafe)

    Eigen::MatrixX<typename Derived::Scalar> matrix_inv = matrix;

    for (unsigned int i = 0; i < matrix_inv.rows(); i++)
    {
        matrix_inv(i, i) = 1 / matrix_inv(i, i);
    }

    return matrix_inv;
}

/// Read a comma-separated numeric file into a matrix of type M, one file line per row.
template <typename M>
M loadCsv(const std::string &path)
{
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line))
    {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ','))
        {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Eigen::Map<const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, Eigen::RowMajor>>(values.data(), rows, values.size() / rows);
}

/**
 * @brief Read the numeric array `root[key]` into `out` (resized if dynamic).
 * @return false, leaving `out` unchanged, if the key is absent.
 * @warning For a fixed-size `out`, the array's length is not checked: a shorter one is
 *          read out of bounds.
 */
template <typename Derived>
bool loadArrayData(Json::Value &root, const std::string &key, Derived &out)
{
    if (root.isMember(key))
    {
        std::vector<typename Derived::Scalar> data;
        for (unsigned int i = 0; i < root[key].size(); i++)
        {
            double value = root[key][i].asDouble();
            data.push_back(static_cast<typename Derived::Scalar>(value));
        }
        if (Derived::RowsAtCompileTime == Eigen::Dynamic || Derived::ColsAtCompileTime == Eigen::Dynamic)
        {
            out.resize(root[key].size(), 1);
        }
        out = Eigen::Map<Derived>(data.data(), out.rows(), out.cols());
        info() << "Loaded array data with key: " << key;
        return true;
    }
    else
    {
        warning() << "Key '" << key << "' not found in JSON. Using default value: " << out;
        return false;
    }
}

/**
 * @brief Read `root[key]`, a 4x4 matrix as 16 numbers in row-major order, into `out`.
 * @return false, leaving `out` unchanged, if the key is absent.
 */
template <typename Derived>
bool loadMatrix4Data(Json::Value &root, const std::string &key, Derived &out)
{
    if (root.isMember(key))
    {
        std::vector<typename Derived::Scalar> data;
        for (unsigned int i = 0; i < root[key].size(); i++)
        {
            double value = root[key][i].asDouble();
            data.push_back(static_cast<typename Derived::Scalar>(value));
        }
        out = Eigen::Map<Derived>(data.data()).transpose();
        info() << "Loaded 4x4 array data with key: " << key;
        return true;
    }
    else
    {
        warning() << "Key '" << key << "' not found in JSON. Using default value: " << out;
        return false;
    }
}

/// Write `in` to `out` as a JSON array (replacing its contents).
template <typename T = double>
void saveArrayData(Json::Value &out, const Eigen::VectorXd &in)
{
    out.clear();
    for (unsigned int i = 0; i < in.size(); i++)
    {
        out.append(in(i));
    }
}

/**
 * @brief Write `in` to `out` as a JSON array of 16 numbers in column-major order.
 * @note Not the inverse of loadMatrix4Data, which reads row-major.
 */
template <typename T = double>
void saveMatrix4dData(Json::Value &out, const Eigen::Matrix4d &in)
{
    out.clear();
    for (unsigned int i = 0; i < 4; i++)
    {
        for (unsigned int j = 0; j < 4; j++)
        {
            out.append(in(j, i));
        }
    }
}

/// Convert a string to T (int, float, double, the long types or std::string). @throws std::runtime_error otherwise.
template <typename T>
T stoX(std::string value)
{
    if constexpr (std::is_same<T, int>::value)
        return std::stoi(value);
    else if constexpr (std::is_same<T, double>::value)
        return std::stod(value);
    else if constexpr (std::is_same<T, float>::value)
        return std::stof(value);
    else if constexpr (std::is_same<T, long>::value)
        return std::stol(value);
    else if constexpr (std::is_same<T, unsigned long>::value)
        return std::stoul(value);
    else if constexpr (std::is_same<T, long long>::value)
        return std::stoll(value);
    else if constexpr (std::is_same<T, unsigned long long>::value)
        return std::stoull(value);
    else if constexpr (std::is_same<T, std::string>::value)
        return value; // No conversion needed for string

    throw std::runtime_error("Unsupported type");
    return T();
}

/// Sequential reader of comma-separated files, by cell or by line.
class CSVReader
{
public:
    /// Open `file_path` for reading.
    CSVReader(std::string file_path) : m_stream(file_path) {};

    /// Skip the rest of the current line. Always returns true.
    bool burnLine();

    /**
     * @brief Read the rest of the current line as cells of type T.
     * @return false if it holds no cell (e.g. at end of file).
     */
    template <typename T = double>
    bool readLine(std::vector<T> &values)
    {
        std::string row;
        std::getline(m_stream, row, '\n');

        values.clear();
        auto start = 0U;
        auto end = row.find(',');
        while (end != std::string::npos)
        {
            values.push_back(stoX<T>(row.substr(start, end - start)));
            start = end + 1;
            end = row.find(',', start);
        }
        end = row.length(); // Handle last character when rows don't end in comma
        std::string substr = row.substr(start, end - start);
        substr.erase(std::remove(substr.begin(), substr.end(), ' '), substr.end());
        if (substr.length() > 0)
        {
            values.push_back(stoX<T>(row.substr(start, end - start)));
        }

        return values.size() > 0;
    }

    /**
     * @brief Read the next cell, up to the next comma (which may span a line end).
     * @return false, leaving `value` unchanged, if the cell is empty (e.g. at end of file).
     */
    template <typename T = unsigned long>
    bool read(T &value)
    {
        std::string cell = "";
        std::getline(m_stream, cell, ',');

        if (cell.size() == 0)
            return false;
        value = stoX<T>(cell);
        return true;
    }

    /// Close the file.
    void close();

private:
    std::ifstream m_stream;
};

/// Print A's block sparsity pattern ('X' nonzero, '-' zero, 'N' NaN), in row_size x col_size blocks.
template <typename Derived>
void printSparsity(const Eigen::MatrixBase<Derived> &A, std::string label, int row_size, int col_size)
{
    std::cout << std::endl
              << "Sparsity pattern of " << label << " (each entry is a " << row_size << "x" << col_size << " block):" << std::endl;
    for (unsigned int i = 0; i < A.rows() / row_size; i++)
    {
        for (unsigned int j = 0; j < A.cols() / col_size; j++)
        {
            Eigen::MatrixX<typename Derived::Scalar> block = A.block(row_size * i, col_size * j, row_size, col_size);

            if (block.hasNaN())
            {
                std::cout << "N ";
            }
            else if (block.isZero())
            {
                std::cout << "- ";
            }
            else
            {
                std::cout << "X ";
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

/// As above, with square blocks of size block_size.
template <typename Derived>
void printSparsity(const Eigen::MatrixBase<Derived> &A, std::string label = "-", int block_size = 6)
{
    printSparsity(A, label, block_size, block_size);
}

/**
 * @brief The selection matrix P that keeps the nonzero rows/columns of the square,
 *        symmetric A (e.g. after removing states from the problem), so that A_r = P^T A P
 *        is, in this project's uses, invertible.
 * @return A.rows() x r, one column per row of A with a nonzero entry.
 */
Eigen::SparseMatrix<double> getFullRankProjection(const Eigen::MatrixX<double> &A);

// namespace lgmath::se3
// {
//     Eigen::Matrix<double, 6, 6> getdJdt(const Eigen::Matrix<double, 18, 1> &gamma, bool dt = true, int num_terms = 10);
// }


/**
 * @brief Read `root[key]` into `out` (a sub-object if T is Json::Value), logging the result.
 * @return false, leaving `out` unchanged, if the key is absent.
 */
template <typename T>
bool safeRead(const Json::Value &root, const std::string &key, T &out)
{
    if (root.isMember(key))
    {
        if constexpr (std::is_same_v<T, Json::Value>)
        {
            out = root[key];
            info() << "Loaded sub-root with key: " << key;
            return true;
        }
        else
        {
            out = root[key].as<T>();
            info() << "Loaded values from key: " << key << " with value: " << out;
            return true;
        }
    }
    else
    {
        warning() << "Key '" << key << "' not found in JSON. Using default value: " << out;
        return false;
    }
}

#endif // UTILITY_FUNCTIONS_H
