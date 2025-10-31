//
// Created by datdau on 10/29/25.
//

#ifndef RECNN_Tracebackable_H
#define RECNN_Tracebackable_H

// Nếu biên dịch bằng NVCC, bật __host__ __device__

#  define HOST_DEVICE __host__ __device__

#include <type_traits>
// iostream chỉ cần cho phần host (operator<<)
#include <iostream>

namespace TommyDat {

    template <typename T>
    struct Tracebackable {
    private:
        T data = 0;
    public:
        short traceBackIDx;
        short traceBackIDy;
        short traceBackIDz;
        // Constructors
        HOST_DEVICE Tracebackable()                 : data(T{}) {}
        HOST_DEVICE Tracebackable(const T& v) : data(v) {}

        // Copy / move default
        HOST_DEVICE Tracebackable(const Tracebackable&) = default;
        HOST_DEVICE Tracebackable(Tracebackable&&) noexcept = default;
        HOST_DEVICE Tracebackable& operator=(const Tracebackable&) = default;
        HOST_DEVICE Tracebackable& operator=(Tracebackable&&) noexcept = default;
        HOST_DEVICE operator T() const { return data; }

        // Gán trực tiếp từ value kiểu T
        HOST_DEVICE Tracebackable& operator=(const T& value) {
            data = value;
            return *this;
        }
        // implicit conversion -> T
        // Getter / Setter
        HOST_DEVICE T get() const { return data; }
        HOST_DEVICE void set(const T& v) { data = v; }

        // Arithmetic operators (member)
        HOST_DEVICE Tracebackable operator+(const Tracebackable& other) const { return Tracebackable(data + other.data); }
        HOST_DEVICE Tracebackable operator-(const Tracebackable& other) const { return Tracebackable(data - other.data); }
        HOST_DEVICE Tracebackable operator*(const Tracebackable& other) const { return Tracebackable(data * other.data); }
        HOST_DEVICE Tracebackable operator/(const Tracebackable& other) const { return Tracebackable(data / other.data); }

        // With scalar T (member)
        HOST_DEVICE Tracebackable operator+(const T& v) const { return Tracebackable(data + v); }
        HOST_DEVICE Tracebackable operator-(const T& v) const { return Tracebackable(data - v); }
        HOST_DEVICE Tracebackable operator*(const T& v) const { return Tracebackable(data * v); }
        HOST_DEVICE Tracebackable operator/(const T& v) const { return Tracebackable(data / v); }

        // Compound assignment
        HOST_DEVICE Tracebackable& operator+=(const Tracebackable& other) { data += other.data; return *this; }
        HOST_DEVICE Tracebackable& operator-=(const Tracebackable& other) { data -= other.data; return *this; }
        HOST_DEVICE Tracebackable& operator*=(const Tracebackable& other) { data *= other.data; return *this; }
        HOST_DEVICE Tracebackable& operator/=(const Tracebackable& other) { data /= other.data; return *this; }

        HOST_DEVICE Tracebackable& operator+=(const T& v) { data += v; return *this; }
        HOST_DEVICE Tracebackable& operator-=(const T& v) { data -= v; return *this; }
        HOST_DEVICE Tracebackable& operator*=(const T& v) { data *= v; return *this; }
        HOST_DEVICE Tracebackable& operator/=(const T& v) { data /= v; return *this; }

        // Comparisons
        // HOST_DEVICE bool operator==(const Tracebackable& other) const { return data == other.data; }
        // HOST_DEVICE bool operator!=(const Tracebackable& other) const { return data != other.data; }
        // HOST_DEVICE bool operator<(const Tracebackable& other) const { return data < other.data; }
        // HOST_DEVICE bool operator<=(const Tracebackable& other) const { return data <= other.data; }
        // HOST_DEVICE bool operator>(const Tracebackable& other) const { return data > other.data; }
        // HOST_DEVICE bool operator>=(const Tracebackable& other) const { return data >= other.data; }

        // Friend declaration for host-only stream output
        friend std::ostream& operator<< (std::ostream& os, const Tracebackable& obj) {
            os << obj.data;
            //   os << obj.traceBackIDx << ":" << obj.traceBackIDy << ":" << obj.traceBackIDz << " ";
        };

    };

    // Non-member operators to support scalar on left: e.g., 2 + Tracebackable<int>
    template <typename T>
    HOST_DEVICE inline Tracebackable<T> operator+(const T& lhs, const Tracebackable<T>& rhs) { return Tracebackable<T>(lhs + rhs.get()); }
    template <typename T>
    HOST_DEVICE inline Tracebackable<T> operator-(const T& lhs, const Tracebackable<T>& rhs) { return Tracebackable<T>(lhs - rhs.get()); }
    template <typename T>
    HOST_DEVICE inline Tracebackable<T> operator*(const T& lhs, const Tracebackable<T>& rhs) { return Tracebackable<T>(lhs * rhs.get()); }
    template <typename T>
    HOST_DEVICE inline Tracebackable<T> operator/(const T& lhs, const Tracebackable<T>& rhs) { return Tracebackable<T>(lhs / rhs.get()); }

} // namespace TommyDat

// Define operator<< only for host (can't use std::ostream on device)
#if !defined(__CUDA_ARCH__)
namespace TommyDat {
    template <typename T>
    inline std::ostream& operator<< (std::ostream& os, const Tracebackable<T>& obj) {
        os << obj.get();
        return os;
    }
}
#endif

#endif // RECNN_Tracebackable_H
