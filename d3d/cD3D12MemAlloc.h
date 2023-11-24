#pragma once
#include <d3d12.h>
#include <dxgi1_4.h>

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C extern
#endif

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Allocator
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

EXTERN_C struct OpaqueAllocator* CreateAllocator(ID3D12Device* pDevice, IDXGIAdapter* pAdapter);

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Allocation
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum ALLOCATION_FLAGS
{
    ALLOCATION_FLAG_NONE = 0,
    ALLOCATION_FLAG_COMMITTED = 0x1,
    ALLOCATION_FLAG_NEVER_ALLOCATE = 0x2,
    ALLOCATION_FLAG_WITHIN_BUDGET = 0x4,
    ALLOCATION_FLAG_UPPER_ADDRESS = 0x8,
    ALLOCATION_FLAG_CAN_ALIAS = 0x10,
    ALLOCATION_FLAG_STRATEGY_MIN_MEMORY = 0x00010000,
    ALLOCATION_FLAG_STRATEGY_MIN_TIME = 0x00020000,
    ALLOCATION_FLAG_STRATEGY_MIN_OFFSET = 0x0004000,
    ALLOCATION_FLAG_STRATEGY_BEST_FIT = ALLOCATION_FLAG_STRATEGY_MIN_MEMORY,
    ALLOCATION_FLAG_STRATEGY_FIRST_FIT = ALLOCATION_FLAG_STRATEGY_MIN_TIME,
    ALLOCATION_FLAG_STRATEGY_MASK =
        ALLOCATION_FLAG_STRATEGY_MIN_MEMORY |
        ALLOCATION_FLAG_STRATEGY_MIN_TIME |
        ALLOCATION_FLAG_STRATEGY_MIN_OFFSET,
};

struct ALLOCATION_DESC
{
    enum ALLOCATION_FLAGS Flags;
    D3D12_HEAP_TYPE HeapType;
    D3D12_HEAP_FLAGS ExtraHeapFlags;
    void* CustomPool;
    void* pPrivateData;
};
struct GpuAllocation {
    void* data0[5];
    ID3D12Resource* resource;
    void* data1[7];
};

EXTERN_C HRESULT CreateResource(
    struct OpaqueAllocator* pAllocator,
    const struct ALLOCATION_DESC* pAllocDesc,
    const D3D12_RESOURCE_DESC* pResourceDesc,
    D3D12_RESOURCE_STATES InitialResourceState,
    const D3D12_CLEAR_VALUE *pOptimizedClearValue,
    struct GpuAllocation** ppAllocation,
    REFIID riidResource,
    void** ppvResource
);

EXTERN_C ULONG ReleaseAllocation(struct GpuAllocation*);
