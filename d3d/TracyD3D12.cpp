//PNC: this is a modified version of TracyD3D12.hpp
//PNC: it's modified to get it to compile with this compiler and to supply a C interface

#include "tracy/Tracy.hpp"
#include "client/TracyProfiler.hpp"
#include "client/TracyCallstack.hpp"

#include <cstdlib>
#include <cassert>
#include <d3d12.h>
#include <dxgi.h>
#include <wrl/client.h>
#include <queue>

namespace tracy {
	struct D3D12QueryPayload
	{
		uint32_t m_queryIdStart = 0;
		uint32_t m_queryCount = 0;
	};

	// Command queue context.
	struct D3D12QueueCtx {
		static constexpr uint32_t MaxQueries = 64 * 1024;  // Queries are begin and end markers, so we can store half as many total time durations. Must be even!

		bool m_initialized = false;

		ID3D12Device* m_device = nullptr;
		ID3D12CommandQueue* m_queue = nullptr;
		uint8_t m_context;
		Microsoft::WRL::ComPtr<ID3D12QueryHeap> m_queryHeap;
		Microsoft::WRL::ComPtr<ID3D12Resource> m_readbackBuffer;

		// In-progress payload.
		uint32_t m_queryLimit = MaxQueries;
		volatile long m_queryCounter = 0;
		uint32_t m_previousQueryCounter = 0;

		uint32_t m_activePayload = 0;
		Microsoft::WRL::ComPtr<ID3D12Fence> m_payloadFence;
		std::queue<D3D12QueryPayload> m_payloadQueue;

		int64_t m_prevCalibration = 0;
		int64_t m_qpcToNs = int64_t{ 1000000000 / GetFrequencyQpc() };

		D3D12QueueCtx(ID3D12Device* device, ID3D12CommandQueue* queue, const char* name, uint16_t name_len)
			: m_device(device)
			, m_queue(queue)
			, m_context(GetGpuCtxCounter().fetch_add(1, std::memory_order_relaxed)) {
			// Verify we support timestamp queries on this queue.

			if (queue->GetDesc().Type == D3D12_COMMAND_LIST_TYPE_COPY) {
				D3D12_FEATURE_DATA_D3D12_OPTIONS3 featureData{};
				bool Success = SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS3, &featureData, sizeof(featureData)));
				assert(Success && featureData.CopyQueueTimestampQueriesSupported && "Platform does not support profiling of copy queues.");
			}

			uint64_t timestampFrequency;

			if (FAILED(queue->GetTimestampFrequency(&timestampFrequency))) {
				assert(false && "Failed to get timestamp frequency.");
			}

			uint64_t cpuTimestamp;
			uint64_t gpuTimestamp;

			if (FAILED(queue->GetClockCalibration(&gpuTimestamp, &cpuTimestamp))) {
				assert(false && "Failed to get queue clock calibration.");
			}

			// Save the device cpu timestamp, not the profiler's timestamp.
			m_prevCalibration = cpuTimestamp * m_qpcToNs;

			cpuTimestamp = Profiler::GetTime();

            const int MY_D3D12_QUERY_HEAP_TYPE_COPY_QUEUE_TIMESTAMP = 5;
			D3D12_QUERY_HEAP_DESC heapDesc{};
			heapDesc.Type = queue->GetDesc().Type == D3D12_COMMAND_LIST_TYPE_COPY ? (D3D12_QUERY_HEAP_TYPE)MY_D3D12_QUERY_HEAP_TYPE_COPY_QUEUE_TIMESTAMP : D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
			heapDesc.Count = m_queryLimit;
			heapDesc.NodeMask = 0;  // #TODO: Support multiple adapters.

			while (FAILED(device->CreateQueryHeap(&heapDesc, IID_PPV_ARGS(&m_queryHeap)))) {
				m_queryLimit /= 2;
				heapDesc.Count = m_queryLimit;
			}

			// Create a readback buffer, which will be used as a destination for the query data.

			D3D12_RESOURCE_DESC readbackBufferDesc{};
			readbackBufferDesc.Alignment = 0;
			readbackBufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
			readbackBufferDesc.Width = m_queryLimit * sizeof(uint64_t);
			readbackBufferDesc.Height = 1;
			readbackBufferDesc.DepthOrArraySize = 1;
			readbackBufferDesc.Format = DXGI_FORMAT_UNKNOWN;
			readbackBufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;  // Buffers are always row major.
			readbackBufferDesc.MipLevels = 1;
			readbackBufferDesc.SampleDesc.Count = 1;
			readbackBufferDesc.SampleDesc.Quality = 0;
			readbackBufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

			D3D12_HEAP_PROPERTIES readbackHeapProps{};
			readbackHeapProps.Type = D3D12_HEAP_TYPE_READBACK;
			readbackHeapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
			readbackHeapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
			readbackHeapProps.CreationNodeMask = 0;
			readbackHeapProps.VisibleNodeMask = 0;  // #TODO: Support multiple adapters.

			if (FAILED(device->CreateCommittedResource(&readbackHeapProps, D3D12_HEAP_FLAG_NONE, &readbackBufferDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_readbackBuffer)))) {
				assert(false && "Failed to create query readback buffer.");
			}

			if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_payloadFence)))) {
				assert(false && "Failed to create payload fence.");
			}

			auto* item = Profiler::QueueSerial();
			MemWrite(&item->hdr.type, QueueType::GpuNewContext);
			MemWrite(&item->gpuNewContext.cpuTime, cpuTimestamp);
			MemWrite(&item->gpuNewContext.gpuTime, gpuTimestamp);
			memset(&item->gpuNewContext.thread, 0, sizeof(item->gpuNewContext.thread));
			MemWrite(&item->gpuNewContext.period, 1E+09f / static_cast<float>(timestampFrequency));
			MemWrite(&item->gpuNewContext.context, m_context);
			MemWrite(&item->gpuNewContext.flags, GpuContextCalibration);
			MemWrite(&item->gpuNewContext.type, GpuContextType::Direct3D12);

#if 0 //def TRACY_ON_DEMAND
			GetProfiler().DeferItem(*item);
#endif

			Profiler::QueueSerialFinish();
			m_initialized = true;

            /*name*/ {
                auto ptr = (char*)tracy_malloc( name_len );
                memcpy( ptr, name, name_len );

                auto item = Profiler::QueueSerial();
                MemWrite( &item->hdr.type, QueueType::GpuContextName );
                MemWrite( &item->gpuContextNameFat.context, m_context );
                MemWrite( &item->gpuContextNameFat.ptr, (uint64_t)ptr );
                MemWrite( &item->gpuContextNameFat.size, name_len );
    #if 0 //def TRACY_ON_DEMAND
                GetProfiler().DeferItem( *item );
    #endif
                Profiler::QueueSerialFinish();
            }
		}

		void NewFrame() {
			uint32_t queryCounter = InterlockedExchange(&m_queryCounter, 0);
			m_payloadQueue.emplace(D3D12QueryPayload{ m_previousQueryCounter, queryCounter });
			m_previousQueryCounter += queryCounter;

			if (m_previousQueryCounter >= m_queryLimit) {
				m_previousQueryCounter -= m_queryLimit;
			}

			m_queue->Signal(m_payloadFence.Get(), ++m_activePayload);
		}

		void Collect() {
			ZoneScopedC(Color::Red4);

#if 0//def TRACY_ON_DEMAND
			if (!GetProfiler().IsConnected()) {
                InterlockedExchange(&m_queryCounter, 0);
				return;
			}
#endif

			// Find out what payloads are available.
			const auto newestReadyPayload = m_payloadFence->GetCompletedValue();
			const auto payloadCount = m_payloadQueue.size() - (m_activePayload - newestReadyPayload);

			if (!payloadCount) {
				return;  // No payloads are available yet, exit out.
			}

			D3D12_RANGE mapRange{ 0, m_queryLimit * sizeof(uint64_t) };

			// Map the readback buffer so we can fetch the query data from the GPU.
			void* readbackBufferMapping = nullptr;

			if (FAILED(m_readbackBuffer->Map(0, &mapRange, &readbackBufferMapping))) {
				assert(false && "Failed to map readback buffer.");
			}

			auto* timestampData = static_cast<uint64_t*>(readbackBufferMapping);
			for (uint32_t i = 0; i < payloadCount; ++i) {
				const auto& payload = m_payloadQueue.front();
				for (uint32_t j = 0; j < payload.m_queryCount; ++j) {
					const auto counter = (payload.m_queryIdStart + j) % m_queryLimit;
					const auto timestamp = timestampData[counter];
					const auto queryId = counter;

					auto* item = Profiler::QueueSerial();
					MemWrite(&item->hdr.type, QueueType::GpuTime);
					MemWrite(&item->gpuTime.gpuTime, timestamp);
					MemWrite(&item->gpuTime.queryId, static_cast<uint16_t>(queryId));
					MemWrite(&item->gpuTime.context, m_context);

					Profiler::QueueSerialFinish();
				}

				m_payloadQueue.pop();
			}

			m_readbackBuffer->Unmap(0, nullptr);

			// Recalibrate to account for drift.

			uint64_t cpuTimestamp;
			uint64_t gpuTimestamp;

			if (FAILED(m_queue->GetClockCalibration(&gpuTimestamp, &cpuTimestamp))) {
				assert(false && "Failed to get queue clock calibration.");
			}

			cpuTimestamp *= m_qpcToNs;

			const auto cpuDelta = cpuTimestamp - m_prevCalibration;
			if (cpuDelta > 0) {
				m_prevCalibration = cpuTimestamp;
				cpuTimestamp = Profiler::GetTime();

				auto* item = Profiler::QueueSerial();
				MemWrite(&item->hdr.type, QueueType::GpuCalibration);
				MemWrite(&item->gpuCalibration.gpuTime, gpuTimestamp);
				MemWrite(&item->gpuCalibration.cpuTime, cpuTimestamp);
				MemWrite(&item->gpuCalibration.cpuDelta, cpuDelta);
				MemWrite(&item->gpuCalibration.context, m_context);
				Profiler::QueueSerialFinish();
			}
		}

		tracy_force_inline uint32_t NextQueryId() {
			uint32_t queryCounter = InterlockedAdd(&m_queryCounter, 2) - 2;
			assert(queryCounter < m_queryLimit && "Submitted too many GPU queries! Consider increasing MaxQueries.");
			const uint32_t id = (m_previousQueryCounter + queryCounter) % m_queryLimit;
			return id;
		}

		tracy_force_inline uint8_t GetId() const {
			return m_context;
		}
	};
}

typedef struct TracyD3D12QueueCtx TracyD3D12QueueCtx;

extern "C" TracyD3D12QueueCtx* tracyCreateD3D12Context(ID3D12Device* device, ID3D12CommandQueue* queue, const char* name, uint16_t name_len) {
    using namespace tracy;
    auto* ctx = static_cast<D3D12QueueCtx*>(tracy_malloc(sizeof(D3D12QueueCtx)));
    new (ctx) D3D12QueueCtx{ device, queue, name, name_len };
    return (TracyD3D12QueueCtx*)ctx;
}

extern "C" void tracyDestroyD3D12Context(TracyD3D12QueueCtx* opaque_ctx) {
    auto ctx = (tracy::D3D12QueueCtx*)opaque_ctx;
    ctx->~D3D12QueueCtx();
    tracy_free(ctx);
}

extern "C" void tracyContextNewFrame(TracyD3D12QueueCtx* opaque_ctx) {
    auto ctx = (tracy::D3D12QueueCtx*)opaque_ctx;
    ctx->NewFrame();
    ctx->Collect();
}

extern "C" uint32_t tracyD3D12ZoneBegin(TracyD3D12QueueCtx* opaque_ctx, const struct ___tracy_source_location_data* cSrcLocation, int depth) {
    tracy::D3D12QueueCtx* ctx = (tracy::D3D12QueueCtx*)opaque_ctx;
    const tracy::SourceLocationData* srcLocation = (const tracy::SourceLocationData*)cSrcLocation;

    using namespace tracy;
    uint32_t queryId = ctx->NextQueryId();
    if (depth > 1) {
        auto* item = Profiler::QueueSerialCallstack(Callstack(depth));
        MemWrite(&item->hdr.type, QueueType::GpuZoneBeginCallstackSerial);
        MemWrite(&item->gpuZoneBegin.cpuTime, Profiler::GetTime());
        MemWrite(&item->gpuZoneBegin.srcloc, reinterpret_cast<uint64_t>(srcLocation));
        MemWrite(&item->gpuZoneBegin.thread, GetThreadHandle());
        MemWrite(&item->gpuZoneBegin.queryId, static_cast<uint16_t>(queryId));
        MemWrite(&item->gpuZoneBegin.context, ctx->GetId());
        Profiler::QueueSerialFinish();
    } else {
        auto* item = Profiler::QueueSerial();
        MemWrite(&item->hdr.type, QueueType::GpuZoneBeginSerial);
        MemWrite(&item->gpuZoneBegin.cpuTime, Profiler::GetTime());
        MemWrite(&item->gpuZoneBegin.srcloc, reinterpret_cast<uint64_t>(srcLocation));
        MemWrite(&item->gpuZoneBegin.thread, GetThreadHandle());
        MemWrite(&item->gpuZoneBegin.queryId, static_cast<uint16_t>(queryId));
        MemWrite(&item->gpuZoneBegin.context, ctx->GetId());
        Profiler::QueueSerialFinish();
    }
	return queryId;
}

extern "C" void tracyD3D12ZoneEnd(TracyD3D12QueueCtx* opaque_ctx, uint32_t query_id) {
    tracy::D3D12QueueCtx* ctx = (tracy::D3D12QueueCtx*)opaque_ctx;
    using namespace tracy;

    auto* item = Profiler::QueueSerial();
    MemWrite(&item->hdr.type, QueueType::GpuZoneEndSerial);
    MemWrite(&item->gpuZoneEnd.cpuTime, Profiler::GetTime());
    MemWrite(&item->gpuZoneEnd.thread, GetThreadHandle());
    MemWrite(&item->gpuZoneEnd.queryId, static_cast<uint16_t>(query_id + 1));
    MemWrite(&item->gpuZoneEnd.context, ctx->GetId());
    Profiler::QueueSerialFinish();
}

extern "C" void tracyD3D12QueryBegin(TracyD3D12QueueCtx* opaque_ctx, ID3D12GraphicsCommandList* cmdList, uint32_t queryId) {
    tracy::D3D12QueueCtx* ctx = (tracy::D3D12QueueCtx*)opaque_ctx;
    cmdList->EndQuery(ctx->m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, queryId);
}

extern "C" void tracyD3D12QueryEnd(TracyD3D12QueueCtx* opaque_ctx, ID3D12GraphicsCommandList* cmdList, uint32_t query_id) {
    tracy::D3D12QueueCtx* ctx = (tracy::D3D12QueueCtx*)opaque_ctx;
    using namespace tracy;
    cmdList->EndQuery(ctx->m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, query_id + 1);
    cmdList->ResolveQueryData(ctx->m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, query_id, 2, ctx->m_readbackBuffer.Get(), query_id * sizeof(uint64_t));
}
