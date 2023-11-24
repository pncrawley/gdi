#ifdef USE_PIX
#include <windows.h>
#include <d3d12.h>
#define _MSC_VER 1900
#include "WinPixEventRuntime/pix3.h"

extern "C" void CPIXInitModule() {
    HMODULE module = PIXLoadLatestWinPixGpuCapturerLibrary();
    if (module == NULL) {
        //TODO: when this fails try to load it from the exe's directory
        OutputDebugStringA("Warning: could not load PIX\n");
        printf("Warning: could not load PIX\n");
    } else {
        OutputDebugStringA("PIX is loaded\n");
        printf("PIX is loaded\n");
    }
}

extern "C" void CPIXSetEnableHUD(BOOL is_enabled) {
    if (is_enabled) {
        PIXSetHUDOptions(PIX_HUD_SHOW_ON_ALL_WINDOWS);
    } else {
        PIXSetHUDOptions(PIX_HUD_SHOW_ON_NO_WINDOWS);
    }
}

extern "C" void CPIXBeginEventCL(ID3D12GraphicsCommandList* cl, BYTE category, char const* name) {
    PIXBeginEvent(cl, PIX_COLOR_INDEX(category), name);
}

extern "C" void CPIXEndEventCL(ID3D12GraphicsCommandList* cl) {
    PIXEndEvent(cl);
}

extern "C" void CPIXBeginEventCQ(ID3D12CommandQueue* cq, BYTE category, char const* name) {
    PIXBeginEvent(cq, PIX_COLOR_INDEX(category), name);
}

extern "C" void CPIXEndEventCQ(ID3D12CommandQueue* cq) {
    PIXEndEvent(cq);
}


extern "C" void CPIXBeginEvent(BYTE category, char const* name) {
    PIXBeginEvent(PIX_COLOR_INDEX(category), name);
}

extern "C" void CPIXEndEvent() {
    PIXEndEvent();
}

extern "C" void CPIXSetMarkerCL(ID3D12GraphicsCommandList* cl, BYTE category, char const* name) {
    PIXSetMarker(cl, PIX_COLOR_INDEX(category), name);
}

extern "C" void CPIXSetMarkerCQ(ID3D12CommandQueue* cq, BYTE category, char const* name) {
    PIXSetMarker(cq, PIX_COLOR_INDEX(category), name);
}

extern "C" void CPIXSetMarker(BYTE category, char const* name) {
    PIXSetMarker(PIX_COLOR_INDEX(category), name);
}


extern "C" void CPIXReportCounter(wchar_t const* name, float value) {
    PIXReportCounter(name, value);
}

#else //USE_PIX

extern "C" void CPIXInitModule() {}
extern "C" void CPIXSetEnableHUD(BOOL is_enabled) {}
extern "C" void CPIXBeginEventCL(ID3D12GraphicsCommandList* cl, BYTE category, char const* name) {}
extern "C" void CPIXEndEventCL(ID3D12GraphicsCommandList* cl) {}
extern "C" void CPIXBeginEventCQ(ID3D12CommandQueue* cq, BYTE category, char const* name) {}
extern "C" void CPIXEndEventCQ(ID3D12CommandQueue* cq) {}
extern "C" void CPIXBeginEvent(BYTE category, char const* name) {}
extern "C" void CPIXEndEvent() {}
extern "C" void CPIXSetMarkerCL(ID3D12GraphicsCommandList* cl, BYTE category, char const* name) {}
extern "C" void CPIXSetMarkerCQ(ID3D12CommandQueue* cq, BYTE category, char const* name) {}
extern "C" void CPIXSetMarker(BYTE category, char const* name) {}
extern "C" void CPIXReportCounter(wchar_t const* name, float value) {}

#endif //USE_PIX