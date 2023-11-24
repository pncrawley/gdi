#include "cD3D12MemAlloc.h"
#define private public //PNC: fuck private
#include "D3D12MemAlloc.h"
#include <assert.h>
#include <stddef.h>
#include <stdio.h>

extern "C" struct OpaqueAllocator* CreateAllocator(ID3D12Device* device, IDXGIAdapter* adapter) {
    //check that our interface matches the real one; we just alias memory for all of these
    //printf("allocation sizes %d != %d\n", sizeof(GpuAllocation), sizeof(D3D12MA::Allocation));
    //printf("offsetof(D3D12MA::Allocation, m_Resource) == %d\n", offsetof(D3D12MA::Allocation, m_Resource));
    //printf("offsetof(GpuAllocation, resource) == %d\n", offsetof(GpuAllocation, resource));
    assert(sizeof(GpuAllocation) == sizeof(D3D12MA::Allocation));
    assert(__alignof(GpuAllocation) == __alignof(D3D12MA::Allocation));
    assert(offsetof(GpuAllocation, resource) == offsetof(D3D12MA::Allocation, m_Resource));
    assert(sizeof(ALLOCATION_DESC) == sizeof(D3D12MA::ALLOCATION_DESC));
    assert(__alignof(ALLOCATION_DESC) == __alignof(D3D12MA::ALLOCATION_DESC));

    D3D12MA::ALLOCATOR_DESC desc = {};
    desc.Flags = (D3D12MA::ALLOCATOR_FLAG_DEFAULT_POOLS_NOT_ZEROED | D3D12MA::ALLOCATOR_FLAG_MSAA_TEXTURES_ALWAYS_COMMITTED);
    desc.pDevice = device;
    desc.pAdapter = adapter;
    D3D12MA::Allocator* allocator = 0;
    HRESULT hr = D3D12MA::CreateAllocator(&desc, &allocator);
    if (hr == S_OK) {
        return (struct OpaqueAllocator*)allocator;
    }
    return 0;
}

extern "C" HRESULT CreateResource(
    struct OpaqueAllocator* pOpaqueAllocator,
    const struct ALLOCATION_DESC* pAllocDesc,
    const D3D12_RESOURCE_DESC* pResourceDesc,
    D3D12_RESOURCE_STATES InitialResourceState,
    const D3D12_CLEAR_VALUE *pOptimizedClearValue,
    struct GpuAllocation** ppAllocation,
    REFIID riidResource,
    void** ppvResource
) {
    assert(ppvResource && ppAllocation);
    D3D12MA::Allocator* pAllocator = (D3D12MA::Allocator*)pOpaqueAllocator;
    HRESULT hr = pAllocator->CreateResource(
        (D3D12MA::ALLOCATION_DESC*)pAllocDesc,
        pResourceDesc,
        InitialResourceState,
        pOptimizedClearValue,
        (D3D12MA::Allocation**) ppAllocation,
        riidResource,
        ppvResource
    );
    assert((*ppvResource) == (void*)((*ppAllocation)->resource));
    return hr;
}

extern "C" ULONG ReleaseAllocation(struct GpuAllocation* alloc) {
    return ((D3D12MA::Allocation*)alloc)->Release();
}
