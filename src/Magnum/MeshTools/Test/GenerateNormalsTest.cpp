/*
    This file is part of Magnum.

    Copyright © 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019
              Vladimír Vondruš <mosra@centrum.cz>

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
*/

#include <sstream>
#include <vector>
#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/ArrayViewStl.h>
#include <Corrade/Containers/StridedArrayView.h>
#include <Corrade/TestSuite/Tester.h>
#include <Corrade/TestSuite/Compare/Container.h>
#include <Corrade/Utility/DebugStl.h>

#include "Magnum/Math/Functions.h"
#include "Magnum/Math/Vector3.h"
#include "Magnum/MeshTools/GenerateNormals.h"
#include "Magnum/Primitives/Cylinder.h"
#include "Magnum/Trade/MeshData3D.h"

namespace Magnum { namespace MeshTools { namespace Test { namespace {

struct GenerateNormalsTest: TestSuite::Tester {
    explicit GenerateNormalsTest();

    void flat();
    #ifdef MAGNUM_BUILD_DEPRECATED
    void flatDeprecated();
    #endif
    void flatWrongCount();
    void flatIntoWrongSize();

    void smoothTwoTriangles();
    void smoothCube();
    void smoothBeveledCube();
    void smoothCylinder();
    void smoothWrongCount();
    void smoothIntoWrongSize();
};

GenerateNormalsTest::GenerateNormalsTest() {
    addTests({&GenerateNormalsTest::flat,
              #ifdef MAGNUM_BUILD_DEPRECATED
              &GenerateNormalsTest::flatDeprecated,
              #endif
              &GenerateNormalsTest::flatWrongCount,
              &GenerateNormalsTest::flatIntoWrongSize,

              &GenerateNormalsTest::smoothTwoTriangles,
              &GenerateNormalsTest::smoothCube,
              &GenerateNormalsTest::smoothBeveledCube,
              &GenerateNormalsTest::smoothCylinder,
              &GenerateNormalsTest::smoothWrongCount,
              &GenerateNormalsTest::smoothIntoWrongSize});
}

/* Two vertices connected by one edge, each wound in another direction */
constexpr Vector3 TwoTriangles[]{
    {-1.0f, 0.0f, 0.0f},
    {0.0f, -1.0f, 0.0f},
    {0.0f, 1.0f, 0.0f},

    {0.0f, -1.0f, 0.0f},
    {0.0f, 1.0f, 0.0f},
    {1.0f, 0.0f, 0.0f}
};

void GenerateNormalsTest::flat() {
    CORRADE_COMPARE_AS(generateFlatNormals(TwoTriangles),
        (Containers::Array<Vector3>{Containers::InPlaceInit, {
            Vector3::zAxis(),
            Vector3::zAxis(),
            Vector3::zAxis(),
            -Vector3::zAxis(),
            -Vector3::zAxis(),
            -Vector3::zAxis()
        }}), TestSuite::Compare::Container);
}

#ifdef MAGNUM_BUILD_DEPRECATED
void GenerateNormalsTest::flatDeprecated() {
    /* Two vertices connected by one edge, each wound in another direction */
    std::vector<UnsignedInt> indices;
    std::vector<Vector3> normals;
    CORRADE_IGNORE_DEPRECATED_PUSH
    std::tie(indices, normals) = MeshTools::generateFlatNormals({
        0, 1, 2,
        1, 2, 3
    }, {
        {-1.0f, 0.0f, 0.0f},
        {0.0f, -1.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {1.0f, 0.0f, 0.0f}
    });
    CORRADE_IGNORE_DEPRECATED_POP

    CORRADE_COMPARE(indices, (std::vector<UnsignedInt>{
        0, 0, 0,
        1, 1, 1
    }));
    CORRADE_COMPARE(normals, (std::vector<Vector3>{
        Vector3::zAxis(),
        -Vector3::zAxis()
    }));
}
#endif

void GenerateNormalsTest::flatWrongCount() {
    std::stringstream out;
    Error redirectError{&out};

    const Vector3 positions[7];
    generateFlatNormals(positions);
    CORRADE_COMPARE(out.str(), "MeshTools::generateFlatNormalsInto(): position count not divisible by 3\n");
}

void GenerateNormalsTest::flatIntoWrongSize() {
    std::stringstream out;
    Error redirectError{&out};

    const Vector3 positions[6];
    Vector3 normals[7];
    generateFlatNormalsInto(positions, normals);
    CORRADE_COMPARE(out.str(), "MeshTools::generateFlatNormalsInto(): bad output size, expected 6 but got 7\n");
}

void GenerateNormalsTest::smoothTwoTriangles() {
    const UnsignedInt indices[]{0, 1, 2, 3, 4, 5};

    /* Should generate the same output as flat normals */
    CORRADE_COMPARE_AS(
        generateSmoothNormals(Containers::stridedArrayView(indices), TwoTriangles),
        (Containers::Array<Vector3>{Containers::InPlaceInit, {
            Vector3::zAxis(),
            Vector3::zAxis(),
            Vector3::zAxis(),
            -Vector3::zAxis(),
            -Vector3::zAxis(),
            -Vector3::zAxis()
        }}), TestSuite::Compare::Container);
}

void GenerateNormalsTest::smoothCube() {
    const Vector3 positions[] {
        {-1.0f, -1.0f,  1.0f},
        { 1.0f, -1.0f,  1.0f},
        { 1.0f,  1.0f,  1.0f},
        {-1.0f,  1.0f,  1.0f},
        {-1.0f,  1.0f, -1.0f},
        { 1.0f,  1.0f, -1.0f},
        { 1.0f, -1.0f, -1.0f},
        {-1.0f, -1.0f, -1.0f},
    };

    const UnsignedByte indices[] {
        0, 1, 2, 0, 2, 3, /* +Z */
        1, 6, 5, 1, 5, 2, /* +X */
        3, 2, 5, 3, 5, 4, /* +Y */
        4, 5, 6, 4, 6, 7, /* -Z */
        3, 4, 7, 3, 7, 0, /* -X */
        7, 6, 1, 7, 1, 0  /* -Y */
    };

    /* Normals should be the same as positions, only normalized */
    CORRADE_COMPARE_AS(
        generateSmoothNormals(Containers::stridedArrayView(indices), positions),
        (Containers::Array<Vector3>{Containers::InPlaceInit, {
            positions[0]/Constants::sqrt3(),
            positions[1]/Constants::sqrt3(),
            positions[2]/Constants::sqrt3(),
            positions[3]/Constants::sqrt3(),
            positions[4]/Constants::sqrt3(),
            positions[5]/Constants::sqrt3(),
            positions[6]/Constants::sqrt3(),
            positions[7]/Constants::sqrt3()
        }}), TestSuite::Compare::Container);
}


constexpr Vector3 BeveledCubePositions[] {
    {-1.0f, -0.6f,  1.1f},
    { 1.0f, -0.6f,  1.1f},
    { 1.0f,  0.6f,  1.1f}, /* +Z */
    {-1.0f,  0.6f,  1.1f},

    { 1.1f, -0.6f,  1.0f},
    { 1.1f, -0.6f, -1.0f},
    { 1.1f,  0.6f, -1.0f}, /* +X */
    { 1.1f,  0.6f,  1.0f},

    {-1.0f,  0.7f,  1.0f},
    { 1.0f,  0.7f,  1.0f},
    { 1.0f,  0.7f, -1.0f}, /* +Y */
    {-1.0f,  0.7f, -1.0f},

    { 1.0f, -0.6f, -1.1f},
    {-1.0f, -0.6f, -1.1f},
    {-1.0f,  0.6f, -1.1f}, /* -Z */
    { 1.0f,  0.6f, -1.1f},

    {-1.0f, -0.7f, -1.0f},
    { 1.0f, -0.7f, -1.0f},
    { 1.0f, -0.7f,  1.0f}, /* -Y */
    {-1.0f, -0.7f,  1.0f},

    {-1.1f, -0.6f, -1.0f},
    {-1.1f, -0.6f,  1.0f},
    {-1.1f,  0.6f,  1.0f}, /* -X */
    {-1.1f,  0.6f, -1.0f}
};

constexpr UnsignedByte BeveledCubeIndices[] {
     0,  1,  2,  0,  2,  3, /* +Z */
     4,  5,  6,  4,  6,  7, /* +X */
     8,  9, 10,  8, 10, 11, /* +Y */
    12, 13, 14, 12, 14, 15, /* -Z */
    16, 17, 18, 16, 18, 19, /* -Y */
    20, 21, 22, 20, 22, 23,  /* -X */

     3,  2,  9,  3,  9,  8, /* +Z / +Y bevel */
     7,  6, 10,  7, 10,  9, /* +X / +Y bevel */
    15, 14, 11, 15, 11, 10, /* -Z / +Y bevel */
    23, 22,  8, 23,  8, 11, /* -X / +Y bevel */

    19, 18,  1, 19,  1,  0, /* -Y / +Z bevel */
    16, 19, 21, 16, 21, 20, /* -Y / -X bevel */
    17, 16, 13, 17, 13, 12, /* -Y / -Z bevel */
    18, 17,  5, 18,  5,  4, /* -Z / +X bevel */

     2,  1,  4,  2,  4,  7, /* +Z / +X bevel */
     6,  5, 12,  6, 12, 15, /* +X / -Z bevel */
    14, 13, 20, 14, 20, 23, /* -Z / -X bevel */
    22, 21,  0, 22,  0,  3, /* -X / +X bevel */

    22,  3,  8, /* -X / +Z / +Y corner */
     2,  7,  9, /* +Z / +X / +Y corner */
     6, 15, 10, /* +X / -Z / +Y corner */
    14, 23, 11, /* -Z / -X / +Y corner */

     0, 21, 19, /* +Z / -X / -Y corner */
    20, 13, 16, /* -X / -Z / -Y corner */
    12,  5, 17, /* -Z / +X / -Y corner */
     4,  1, 18  /* +X / +Z / -Y corner */
};

void GenerateNormalsTest::smoothBeveledCube() {
    /* Data taken from Primitives::cubeSolid() and expanded a bit, with bevel
       faces added */

    /* Normals should be mirrored on the X/Y/Z plane and with a circular
       symmetry around the Y axis, signs corresponding to position signs. */
    Vector3 z{0.0462723f, 0.0754969f, 0.996072f};
    Vector3 x{0.996072f, 0.0754969f, 0.0462723f};
    Vector3 y{0.0467958f, 0.997808f, 0.0467958f};
    CORRADE_COMPARE_AS(generateSmoothNormals(
        Containers::stridedArrayView(BeveledCubeIndices), BeveledCubePositions),
        (Containers::Array<Vector3>{Containers::InPlaceInit, {
            z*Math::sign(BeveledCubePositions[ 0]),
            z*Math::sign(BeveledCubePositions[ 1]),
            z*Math::sign(BeveledCubePositions[ 2]), /* +Z */
            z*Math::sign(BeveledCubePositions[ 3]),

            x*Math::sign(BeveledCubePositions[ 4]),
            x*Math::sign(BeveledCubePositions[ 5]),
            x*Math::sign(BeveledCubePositions[ 6]), /* +X */
            x*Math::sign(BeveledCubePositions[ 7]),

            y*Math::sign(BeveledCubePositions[ 8]),
            y*Math::sign(BeveledCubePositions[ 9]),
            y*Math::sign(BeveledCubePositions[10]), /* +Y */
            y*Math::sign(BeveledCubePositions[11]),

            z*Math::sign(BeveledCubePositions[12]),
            z*Math::sign(BeveledCubePositions[13]),
            z*Math::sign(BeveledCubePositions[14]), /* -Z */
            z*Math::sign(BeveledCubePositions[15]),

            y*Math::sign(BeveledCubePositions[16]),
            y*Math::sign(BeveledCubePositions[17]),
            y*Math::sign(BeveledCubePositions[18]), /* -Y */
            y*Math::sign(BeveledCubePositions[19]),

            x*Math::sign(BeveledCubePositions[20]),
            x*Math::sign(BeveledCubePositions[21]),
            x*Math::sign(BeveledCubePositions[22]), /* -X */
            x*Math::sign(BeveledCubePositions[23])
        }}), TestSuite::Compare::Container);
}

void GenerateNormalsTest::smoothCylinder() {
    const Trade::MeshData3D data = Primitives::cylinderSolid(1, 5, 1.0f);

    /* Output should be exactly the same as the cylinder normals */
    CORRADE_COMPARE_AS(Containers::arrayView(generateSmoothNormals(
        Containers::stridedArrayView(data.indices()),
        Containers::stridedArrayView(data.positions(0)))),
        Containers::arrayView(data.normals(0)), TestSuite::Compare::Container);
}

void GenerateNormalsTest::smoothWrongCount() {
    std::stringstream out;
    Error redirectError{&out};

    const UnsignedByte indices[7]{};
    const Vector3 positions[1];
    generateSmoothNormals(Containers::stridedArrayView(indices), positions);
    CORRADE_COMPARE(out.str(), "MeshTools::generateSmoothNormalsInto(): index count not divisible by 3\n");
}

void GenerateNormalsTest::smoothIntoWrongSize() {
    std::stringstream out;
    Error redirectError{&out};

    const UnsignedByte indices[6]{};
    const Vector3 positions[3];
    Vector3 normals[4];
    generateSmoothNormalsInto(Containers::stridedArrayView(indices), positions, normals);
    CORRADE_COMPARE(out.str(), "MeshTools::generateSmoothNormalsInto(): bad output size, expected 3 but got 4\n");
}

}}}}

CORRADE_TEST_MAIN(Magnum::MeshTools::Test::GenerateNormalsTest)