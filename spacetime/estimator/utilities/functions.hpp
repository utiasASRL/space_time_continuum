/**
 * @file functions.hpp
 * @brief Small global-namespace helpers: matrix checks and inverses, CSV and JSON
 *        reading/writing (used by Utilities::ConfigLoader and the dataloaders), and
 *        debugging printouts.
 */

#ifndef UTILITY_FUNCTIONS_H
#define UTILITY_FUNCTIONS_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <json/json.h>
#include <json/value.h>
#include <gtsam/nonlinear/Symbol.h>

#include <spacetime/estimator/utilities/logger.hpp>

/// Millidegrees to radians (pi / 180000).
#define MDEG2RAD 0.0000174532925
/// Library-wide tolerance on time and arclength offsets: two times (or arclengths) closer than this are the same.
#define TOLERANCE 1e-8

/// A rows x rows zero matrix.
inline Eigen::MatrixXd ZERO(int rows) { return Eigen::MatrixXd::Zero(rows, rows); }
/// A rows x rows identity matrix.
inline Eigen::MatrixXd EYE(int rows) { return Eigen::MatrixXd::Identity(rows, rows); }

/**
 * @brief Assert (when NDEBUG is not defined) that the 3x3 matrix C is orthonormal, to 1e-5
 *        in every entry of C^T C - I. The determinant's sign is not checked.
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

/// Assert (when NDEBUG is not defined) that the 4x4 matrix T is in SE(3): an orthonormal top-left block (see validateRotationMatrix) and a bottom row [0 0 0 1] to 1e-10.
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

/// Inverse of the SE(3) matrix T = [C, r; 0, 1], i.e. [C^T, -C^T r; 0, 1] (asserts T is in SE(3), see validateTransformationMatrix).
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

/// Assert (when NDEBUG is not defined) that `matrix` is square and diagonal, to 1e-10.
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

/// Inverse of a diagonal matrix (element-wise reciprocal of the diagonal; asserts it is diagonal, see validateDiagonalMatrix). A zero diagonal entry gives inf.
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

/**
 * @brief Read a comma-separated numeric file into a matrix of type M, one non-blank file line per row.
 * @tparam M An Eigen matrix type. If it has a fixed size, the file must match it.
 * @throws std::runtime_error if the file cannot be opened, has no rows, has rows of different
 *         lengths, or does not match a fixed-size M.
 * @throws std::invalid_argument (from std::stod) for a non-numeric cell.
 */
template <typename M>
M loadCsv(const std::string &path)
{
    std::ifstream indata(path);
    if (!indata.is_open())
        throw std::runtime_error("loadCsv: cannot open '" + path + "'.");
    std::string line;
    std::vector<double> values;
    std::size_t rows = 0, cols = 0;
    while (std::getline(indata, line))
    {
        if (line.find_first_not_of(" \t\r") == std::string::npos)
            continue; // blank line (e.g. a trailing newline)
        std::stringstream lineStream(line);
        std::string cell;
        std::size_t row_cols = 0;
        while (std::getline(lineStream, cell, ','))
        {
            values.push_back(std::stod(cell));
            ++row_cols;
        }
        if (rows == 0)
            cols = row_cols;
        else if (row_cols != cols)
            throw std::runtime_error("loadCsv: '" + path + "' line " + std::to_string(rows + 1) + " has " + std::to_string(row_cols) +
                                     " cells, the first has " + std::to_string(cols) + ".");
        ++rows;
    }
    if (rows == 0)
        throw std::runtime_error("loadCsv: '" + path + "' has no rows.");
    if ((M::RowsAtCompileTime != Eigen::Dynamic && static_cast<std::size_t>(M::RowsAtCompileTime) != rows) ||
        (M::ColsAtCompileTime != Eigen::Dynamic && static_cast<std::size_t>(M::ColsAtCompileTime) != cols))
        throw std::runtime_error("loadCsv: '" + path + "' is " + std::to_string(rows) + " x " + std::to_string(cols) +
                                 ", which does not fit the fixed-size matrix type.");
    return Eigen::Map<const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, Eigen::RowMajor>>(values.data(), rows, cols);
}

/// The numbers of the JSON array `value` (named `key` in errors).
/// @throws std::invalid_argument if `value` is not an array, or holds a non-numeric entry.
template <typename Scalar>
std::vector<Scalar> readNumericArray(const Json::Value &value, const std::string &key)
{
    if (!value.isArray())
        throw std::invalid_argument("Key '" + key + "' must be a JSON array of numbers.");
    std::vector<Scalar> data;
    data.reserve(value.size());
    for (Json::ArrayIndex i = 0; i < value.size(); i++)
    {
        if (!value[i].isNumeric())
            throw std::invalid_argument("Key '" + key + "' entry " + std::to_string(i) + " is not a number.");
        data.push_back(static_cast<Scalar>(value[i].asDouble()));
    }
    return data;
}

/**
 * @brief Read the numeric array `root[key]` into `out`, in Eigen's storage order of `Derived`.
 *        A fixed-size `out` must match the array's length. A dynamic one is resized: a vector
 *        to the array's length, a matrix with one fixed dimension to fit it, and a fully
 *        dynamic matrix to a column.
 * @return false, leaving `out` unchanged (and logging a warning), if the key is absent.
 * @throws std::invalid_argument if `root[key]` is not a numeric array, or its length does not fit `out`.
 */
template <typename Derived>
bool loadArrayData(Json::Value &root, const std::string &key, Derived &out)
{
    if (!root.isMember(key))
    {
        warning() << "Key '" << key << "' not found in JSON. Using default value: " << out;
        return false;
    }
    std::vector<typename Derived::Scalar> data = readNumericArray<typename Derived::Scalar>(root[key], key);
    const Eigen::Index n = static_cast<Eigen::Index>(data.size());
    constexpr int kRows = Derived::RowsAtCompileTime, kCols = Derived::ColsAtCompileTime;

    Eigen::Index rows = n, cols = 1;
    if constexpr (kRows != Eigen::Dynamic && kCols != Eigen::Dynamic)
        rows = kRows, cols = kCols;
    else if constexpr (kRows == 1)
        rows = 1, cols = n;
    else if constexpr (kRows != Eigen::Dynamic)
        rows = kRows, cols = n / kRows;
    else if constexpr (kCols != Eigen::Dynamic)
        rows = n / kCols, cols = kCols;
    if (rows * cols != n)
        throw std::invalid_argument("Key '" + key + "' has " + std::to_string(n) + " entries, which do not fit a " +
                                    (kRows == Eigen::Dynamic ? std::string("N") : std::to_string(kRows)) + " x " +
                                    (kCols == Eigen::Dynamic ? std::string("N") : std::to_string(kCols)) + " matrix.");

    out.resize(rows, cols);
    out = Eigen::Map<Derived>(data.data(), rows, cols);
    info() << "Loaded array data with key: " << key;
    return true;
}

/**
 * @brief Read `root[key]`, a 4x4 matrix as 16 numbers in row-major order, into `out`.
 * @tparam Derived A fixed-size, column-major 4x4 Eigen matrix (e.g. Eigen::Matrix4d).
 * @return false, leaving `out` unchanged, if the key is absent.
 * @throws std::invalid_argument if `root[key]` is not an array of exactly 16 numbers.
 */
template <typename Derived>
bool loadMatrix4Data(Json::Value &root, const std::string &key, Derived &out)
{
    static_assert(Derived::RowsAtCompileTime == 4 && Derived::ColsAtCompileTime == 4, "loadMatrix4Data reads a fixed-size 4x4 matrix.");
    if (!root.isMember(key))
    {
        warning() << "Key '" << key << "' not found in JSON. Using default value: " << out;
        return false;
    }
    std::vector<typename Derived::Scalar> data = readNumericArray<typename Derived::Scalar>(root[key], key);
    if (data.size() != 16)
        throw std::invalid_argument("Key '" + key + "' must hold 16 numbers (a row-major 4x4 matrix), not " + std::to_string(data.size()) + ".");
    out = Eigen::Map<Derived>(data.data()).transpose();
    info() << "Loaded 4x4 array data with key: " << key;
    return true;
}

/// Write `in` to `out` as a JSON array (replacing its contents). `T` is unused.
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

/**
 * @brief Convert a string to T (int, float, double, long, unsigned long, long long,
 *        unsigned long long, or std::string, returned as is).
 * @throws std::runtime_error for any other T; std::invalid_argument or std::out_of_range
 *         (from std::sto*) for a string that does not parse.
 */
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

/// Sequential reader of comma-separated files, by cell or by line. A file that fails to open is not reported: every read then returns false.
class CSVReader
{
public:
    /// Open `file_path` for reading.
    CSVReader(std::string file_path) : m_stream(file_path) {};

    /// Skip the rest of the current line. Always returns true.
    bool burnLine();

    /**
     * @brief Read the rest of the current line as cells of type T (see stoX), replacing
     *        `values`. A trailing comma or whitespace-only last cell is ignored.
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

/// Print A's block sparsity pattern to stdout ('X' nonzero, '-' zero, 'N' any NaN), in row_size x col_size blocks; trailing partial blocks are not printed.
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


/**
 * @brief Read `root[key]` into `out` (a sub-object if T is Json::Value), logging the result.
 * @tparam T Json::Value, or any type Json::Value::as<T>() supports that can be streamed.
 * @return false, leaving `out` unchanged (and logging a warning), if the key is absent.
 * @throws Json::LogicError if the value cannot be converted to T.
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
