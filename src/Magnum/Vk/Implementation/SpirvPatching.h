/*
    This file is part of Magnum.

    Copyright © 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
                2020, 2021 Vladimír Vondruš <mosra@centrum.cz>

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

#include <Corrade/Containers/Array.h>

#include "Magnum/Magnum.h"
#include "Magnum/Math/Functions.h"
#include "MagnumExternal/Vulkan/spirv.h"

namespace Magnum { namespace Vk { namespace Implementation { namespace {

bool isSpirv(const UnsignedInt* code, UnsignedInt size) {
    return size >= 5 && code[0] == SpvMagicNumber;
}

bool spirvPatchSwiftShaderConflictingMultiEntrypointLocations(Containers::ArrayView<UnsignedInt> data) {
    /* Skip the header, assuming it's valid */
    data = data.suffix(5);

    const auto find = [&data](const SpvOp op) -> Containers::ArrayView<UnsignedInt> {
        Containers::ArrayView<UnsignedInt> dataIteration = data;

        while(!dataIteration.empty()) {
            const UnsignedInt instructionSize = dataIteration[0] >> 16;
            if(dataIteration.size() < instructionSize) {
                data = dataIteration;
                return nullptr;
            }

            const Containers::ArrayView<UnsignedInt> instruction = dataIteration.prefix(instructionSize);
            dataIteration = dataIteration.suffix(instructionSize);
            if((instruction[0] & 0xffffu) == op) {
                data = dataIteration;
                return instruction;
            }
        }

        return nullptr;
    };

    /* Get vertex and fragment entrypoints. Those are always first. */
    Containers::ArrayView<UnsignedInt> vertexEntryPoint, fragmentEntryPoint;
    while(const Containers::ArrayView<UnsignedInt> entryPoint = find(SpvOpEntryPoint)) {
        /* Expecting at least op, execution model, ID, name. If less, it's an
           invalid SPIR-V. */
        if(entryPoint.size() < 4) return false;

        if(entryPoint[1] == SpvExecutionModelVertex)
            vertexEntryPoint = entryPoint;
        if(entryPoint[1] == SpvExecutionModelFragment)
            fragmentEntryPoint = entryPoint;
    }

    /* If there aren't both, this bug doesn't affect the shader. */
    if(!vertexEntryPoint || !fragmentEntryPoint) return false;

    /* Find in/out IDs for both, which are after the null-terminated name. */
    Containers::ArrayView<UnsignedInt> vertexEntryPointIds, fragmentEntryPointIds;
    for(std::size_t i = 3; i != vertexEntryPoint.size(); ++i) {
        if(vertexEntryPoint[i] >> 24 == 0) {
            vertexEntryPointIds = vertexEntryPoint.suffix(i + 1);
            break;
        }
    }
    for(std::size_t i = 3; i != fragmentEntryPoint.size(); ++i) {
        if(fragmentEntryPoint[i] >> 24 == 0) {
            fragmentEntryPointIds = fragmentEntryPoint.suffix(i + 1);
            break;
        }
    }

    /* If there aren't inputs/outputs for either, the shader is weird. And also
       the bug doesn't affect it. */
    if(vertexEntryPointIds.empty() || fragmentEntryPointIds.empty())
        return false;

    struct EntryPointInterface {
        UnsignedInt* location{};
        SpvStorageClass storageClass{};
    };
    /** @todo DynamicArray, once it materizalizes */
    Containers::Array<EntryPointInterface> vertexInterface{vertexEntryPointIds.size()};
    Containers::Array<EntryPointInterface> fragmentInterface{fragmentEntryPointIds.size()};

    /* Find locations for those, and remember the max location index so we can
       use the ones after. Decorations are always after entrypoints. */
    UnsignedInt maxLocation = 0;
    while(const Containers::ArrayView<UnsignedInt> decoration = find(SpvOpDecorate)) {
        /* Expecting at least op, ID, SpvDecorationLocation, location. The
           instruction can be three words, so if we get less than 4 it's not an
           error. */
        if(decoration.size() < 4 || decoration[2] != SpvDecorationLocation)
            continue;

        maxLocation = Math::max(maxLocation, decoration[3]);

        for(std::size_t i = 0; i != vertexEntryPointIds.size(); ++i) {
            if(decoration[1] == vertexEntryPointIds[i]) {
                vertexInterface[i].location = decoration + 3;
                break;
            }
        }
        for(std::size_t i = 0; i != fragmentEntryPointIds.size(); ++i) {
            if(decoration[1] == fragmentEntryPointIds[i]) {
                fragmentInterface[i].location = decoration + 3;
                break;
            }
        }
    }

    /* Find out storage classes for all these */
    while(const Containers::ArrayView<UnsignedInt> variable = find(SpvOpVariable)) {
        /* Expecting at least op, result, ID, SpvStorageClass. If less, it's an
           invalid SPIR-V. */
        if(variable.size() < 4) return false;

        for(std::size_t i = 0; i != vertexEntryPointIds.size(); ++i) {
            if(variable[2] == vertexEntryPointIds[i]) {
                vertexInterface[i].storageClass = SpvStorageClass(variable[3]);
                break;
            }
        }
        for(std::size_t i = 0; i != fragmentEntryPointIds.size(); ++i) {
            if(variable[2] == fragmentEntryPointIds[i]) {
                fragmentInterface[i].storageClass = SpvStorageClass(variable[3]);
                break;
            }
        }
    }

    /* For all vertex outputs check if there are fragment outputs with the same
       locations */
    for(const EntryPointInterface& vertexOutput: vertexInterface) {
        /* Ignore what's not an output or what doesn't have a location (for
           example a builtin) */
        if(vertexOutput.storageClass != SpvStorageClassOutput || !vertexOutput.location)
            continue;

        for(const EntryPointInterface& fragmentOutput: fragmentInterface) {
            /* Ignore what's not an output or what doesn't have a location (for
               example a builtin) */
            if(fragmentOutput.storageClass != SpvStorageClassOutput || !fragmentOutput.location) continue;

            /* The same location used, we need to remap. Use the next highest
               unused location and change also the corresponding fragment
               input, if there's any. */
            if(*vertexOutput.location == *fragmentOutput.location) {
                const UnsignedInt newLocation = ++maxLocation;
                for(const EntryPointInterface& fragmentInput: fragmentInterface) {
                    if(fragmentInput.storageClass != SpvStorageClassInput)
                        continue;
                    if(*fragmentInput.location == *vertexOutput.location) {
                        *fragmentInput.location = newLocation;
                        break;
                    }
                }
                *vertexOutput.location = newLocation;
                break;
            }
        }
    }

    return true;
}

}}}}
