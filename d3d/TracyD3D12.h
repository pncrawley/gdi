#include <d3d12.h>

#ifdef TRACY_ENABLE

typedef struct TracyD3D12QueueCtx TracyD3D12QueueCtx;
extern TracyD3D12QueueCtx* tracyCreateD3D12Context(ID3D12Device*, ID3D12CommandQueue*, const char* name, uint16_t name_len);
extern void tracyDestroyD3D12Context(TracyD3D12QueueCtx*);
extern void tracyContextNewFrame(TracyD3D12QueueCtx*);

extern uint32_t tracyD3D12ZoneBegin(TracyD3D12QueueCtx* opaque_ctx, const struct ___tracy_source_location_data* cSrcLocation, int depth);
extern void tracyD3D12ZoneEnd(TracyD3D12QueueCtx* opaque_ctx, uint32_t query_id);
extern void tracyD3D12QueryBegin(TracyD3D12QueueCtx* opaque_ctx, ID3D12GraphicsCommandList* cmdList, uint32_t query_id);
extern void tracyD3D12QueryEnd(TracyD3D12QueueCtx* opaque_ctx, ID3D12GraphicsCommandList* cmdList, uint32_t query_id);

#endif //TRACY_ENABLE
